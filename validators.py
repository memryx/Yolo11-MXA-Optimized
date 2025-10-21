import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from memryx import SyncAccl  # type:ignore
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import LOGGER, TQDM

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def get_accl(model_path):
    dfp_path = model_path.with_suffix(".dfp")
    accl = SyncAccl(dfp_path, local_mode=True)
    return accl


def get_ort(model_path):
    post_path = str(model_path).replace(".pt", "_post.onnx")
    return ort.InferenceSession(post_path)


# NOTE: The three classes below differ only in what they inherit from, and
# in their usage of mxa_detect, mxa_segment, and mxa_pose methods.


class MxaDetectionValidator(DetectionValidator):
    """
    The Validator must be a child of BaseValidator which is the parent
    of DetectionValidator. The BaseValidator defines the __call__
    method which we need to override.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set required attributes
        self.stride = 32
        self.training = False

        model_path = Path(self.args.model)
        self.model_name = model_path.stem

        # Ensure your paths/naming scheme matches
        self.mxa = get_accl(model_path)
        self.ort = get_ort(model_path)

    def __call__(self, model):  # type:ignore
        model.eval()

        # Create COCO dataloader
        self.data = check_det_dataset(self.args.data)
        self.dataloader = self.get_dataloader(
            self.data.get(self.args.split), self.args.batch
        )

        # Validation Loop
        self.init_metrics((model))
        self.jdict = []
        progress_bar = TQDM(
            self.dataloader,
            desc=self.get_desc(),
            total=len(self.dataloader),
            maxinterval=600,
            miniters=1 if not os.getenv("CI") == "true" else 10,
        )
        for batch in progress_bar:
            if len(batch["img"]) != self.args.batch:
                break
            batch = self.preprocess(batch)
            preds = self.mxa_detect(batch["img"])
            preds = self.postprocess(preds)
            self.update_metrics(preds, batch)

        # Compute and print stats
        stats = self.get_stats()
        self.check_stats(stats)
        self.finalize_metrics()
        self.print_results()

        # Save predictions and evaluate on pycocotools
        with open(str(self.save_dir / "predictions.json"), "w") as f:
            LOGGER.info(f"Saving {f.name}...")
            json.dump(self.jdict, f)
        stats = self.eval_json(stats)

        # Bad hack
        if isinstance(self.mxa, SyncAccl):
            self.mxa.shutdown()  # type:ignore

        return stats

    def mxa_detect(self, batch):
        """
        Detection using MXA accelerator.

        Args:
            img (torch.Tensor): Input image. (B, 3, 640, 640)

        Returns:
            preds (list): List of length 2.
                preds[0] (torch.Tensor): Predictions. (B, 84, 8400)
                preds[1] (None): Unused fmaps
        Notes:
            Fj in (64, 80) and Fi in (80, 40, 20)
        """
        # Pass images through accelerator
        batch = batch.detach().cpu().numpy()  # (B, 3, H, W)
        batch = [img[np.newaxis, ...] for img in batch]  # (B, 1, 3, H, W)
        accl_out = self.mxa.run(batch)  # (B, 6, 1, Fj, Fi, Fi)

        # Process accl out for onnxruntime
        if isinstance(accl_out[0], np.ndarray):
            accl_out = [accl_out]

        onnx_inps = []  # 6, B, Fj, Fi, Fi
        batch_size, num_inputs = len(accl_out), len(accl_out[0])  # B, 6
        for inp_idx in range(num_inputs):
            inp = [
                accl_out[batch_idx][inp_idx].squeeze(axis=0)
                for batch_idx in range(batch_size)
            ]  # B, Fj, Fi, Fi
            onnx_inps.append(inp)

        onnx_inp_names = [inp.name for inp in self.ort.get_inputs()]
        input_feed = {k: v for k, v in zip(onnx_inp_names, onnx_inps)}

        # Pass fmaps through onnxruntime
        onnx_out = self.ort.run(None, input_feed)
        out = torch.from_numpy(onnx_out[0])  # (B, 84, 8400)

        preds = [out, None]
        return preds


class MxaSegmentationValidator(SegmentationValidator):
    """
    The Validator must be a child of BaseValidator which is the parent
    of SegmentationValidator. The BaseValidator defines the __call__
    method which we need to override.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set required attributes
        self.stride = 32
        self.training = False
        self.args.plots = False

        model_path = Path(self.args.model)
        self.model_name = model_path.stem

        # Ensure your paths/naming scheme matches
        self.mxa = get_accl(model_path)
        self.ort = get_ort(model_path)

    def __call__(self, model):  # type:ignore
        model.eval()

        # Create COCO dataloader
        self.data = check_det_dataset(self.args.data)
        self.dataloader = self.get_dataloader(
            self.data.get(self.args.split), self.args.batch
        )

        # Validation Loop
        self.init_metrics((model))
        self.jdict = []
        progress_bar = TQDM(
            self.dataloader,
            desc=self.get_desc(),
            total=len(self.dataloader),
            maxinterval=600,
            miniters=1 if not os.getenv("CI") == "true" else 10,
        )
        for i, batch in enumerate(progress_bar):
            if len(batch["img"]) != self.args.batch:
                break
            self.batch_i = i  # For plots
            batch = self.preprocess(batch)
            preds = self.mxa_segment(batch["img"])
            preds = self.postprocess(preds)
            self.update_metrics(preds, batch)

        # Compute and print stats
        stats = self.get_stats()
        self.check_stats(stats)
        self.finalize_metrics()
        self.print_results()

        # Save predictions and evaluate on pycocotools
        with open(str(self.save_dir / "predictions.json"), "w") as f:
            LOGGER.info(f"Saving {f.name}...")
            json.dump(self.jdict, f)
        stats = self.eval_json(stats)

        # Bad hack
        if isinstance(self.mxa, SyncAccl):
            self.mxa.shutdown()  # type:ignore

        return stats

    def mxa_segment(self, batch):
        """
        Segmentation using MXA accelerator.

        Args:
            batch (torch.Tensor): Input image. (B, 3, 640, 640)

        Returns:
            preds (list): List of length 2.
                preds[0] (torch.Tensor): Boxes (B, 116, 8400)
                preds[1] (torch.Tensor): Masks(B, 32, 160, 160)

        Notes:
            For shapes: Fj in (64, 80) and Fi in (80, 40, 20)
        """
        # Pass images through accelerator
        batch = batch.detach().cpu().numpy()  # (B, 3, 640, 640)
        batch = [img[np.newaxis, ...] for img in batch]  # (B, 1, 3, H, W)
        accl_out = self.mxa.run(batch)  # (B, 10, 1, Fj, Fi, Fi)

        # Prepare accelerator output as input to onnx post-processor
        if isinstance(accl_out[0], np.ndarray):
            accl_out = [accl_out]

        onnx_inps = []  # 10, B, Fj, Fi, Fi
        batch_size, num_inputs = len(accl_out), len(accl_out[0])  # B, 10
        for inp_idx in range(num_inputs):
            inp = np.array(
                [
                    accl_out[batch_idx][inp_idx].squeeze(axis=0)
                    for batch_idx in range(batch_size)
                ]
            )  # B, Fj, Fi, Fi
            onnx_inps.append(inp)

        # Reorder names and fmaps to match accl output order
        onnx_inp_names = [inp.name for inp in self.ort.get_inputs()]
        input_feed = {}

        if "yolo11" in self.model_name:
            onnx_inp_names.insert(0, onnx_inp_names.pop(1))
            onnx_inp_names.insert(2, onnx_inp_names.pop(5))
            onnx_inp_names.insert(5, onnx_inp_names.pop(6))
            onnx_inp_names.insert(7, onnx_inp_names.pop(8))

        if "yolov8" in self.model_name:
            onnx_inp_names.insert(3, onnx_inp_names.pop(7))
            onnx_inp_names.insert(6, onnx_inp_names.pop(8))

        input_feed = {name: fmap for name, fmap in zip(onnx_inp_names, onnx_inps)}

        # Pass fmaps through onnxruntime
        onnx_out = self.ort.run(None, input_feed)
        preds = [
            torch.from_numpy(onnx_out[1]),  # Boxes (B, 116, 8400)
            torch.from_numpy(onnx_out[0]),  # Masks (B, 32, 160, 160)
        ]
        return preds


class MxaPoseValidator(PoseValidator):
    """
    The Validator must be a child of BaseValidator which is the parent
    of PoseValidator. The BaseValidator defines the __call__ method
    which we need to override.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set required attributes
        self.stride = 32
        self.training = False

        model_path = Path(self.args.model)
        self.model_name = model_path.stem

        # Ensure your paths/naming scheme matches
        self.mxa = get_accl(model_path)
        self.ort = get_ort(model_path)

    def __call__(self, model):  # type:ignore
        model.eval()

        # Create COCO dataloader
        self.data = check_det_dataset(self.args.data)
        self.dataloader = self.get_dataloader(
            self.data.get(self.args.split), self.args.batch
        )

        # Validation Loop
        self.init_metrics((model))
        self.jdict = []
        progress_bar = TQDM(
            self.dataloader,
            desc=self.get_desc(),
            total=len(self.dataloader),
            maxinterval=600,
            miniters=1 if not os.getenv("CI") == "true" else 10,
        )
        for batch in progress_bar:
            if len(batch["img"]) != self.args.batch:
                break
            batch = self.preprocess(batch)
            preds = self.mxa_pose(batch["img"])
            preds = self.postprocess(preds)
            self.update_metrics(preds, batch)

        # Compute and print stats
        stats = self.get_stats()
        self.check_stats(stats)
        self.finalize_metrics()
        self.print_results()

        # Save predictions and evaluate on pycocotools
        with open(str(self.save_dir / "predictions.json"), "w") as f:
            LOGGER.info(f"Saving {f.name}...")
            json.dump(self.jdict, f)
        stats = self.eval_json(stats)

        # Bad hack
        if isinstance(self.mxa, SyncAccl):
            self.mxa.shutdown()  # type:ignore

        return stats

    def mxa_pose(self, batch):
        """
        Pose Estimation using MXA accelerator.

        Args:
            batch (torch.Tensor): Input image. (B, 3, 640, 640)

        Returns:
            preds (list): List of length 2.
                preds[0] (torch.Tensor): Predictions. (B, 56, 8400)
                preds[1] (None): Unused loss output
        """
        # Pass images through accelerator
        batch = batch.detach().cpu().numpy()  # (B, 3, H, W)
        batch = [img[np.newaxis, ...] for img in batch]  # (B, 1, 3, H, W)
        accl_out = self.mxa.run(batch)  # (B, 9, 1, Fj, Fi, Fi)

        # Prepare accelerator output as input to onnx post-processor
        if isinstance(accl_out[0], np.ndarray):
            accl_out = [accl_out]

        onnx_inps = []  # 9, B, Fj, Fi, Fi
        batch_size, num_inputs = len(accl_out), len(accl_out[0])  # B, 9
        for inp_idx in range(num_inputs):
            inp = np.array(
                [
                    accl_out[batch_idx][inp_idx].squeeze(axis=0)
                    for batch_idx in range(batch_size)
                ]
            )  # B, Fj, Fi, Fi
            onnx_inps.append(inp)

        # Reorder names and fmaps to match accl output order
        onnx_inp_names = [inp.name for inp in self.ort.get_inputs()]

        if "yolo11" in self.model_name:
            onnx_inp_names.insert(1, onnx_inp_names.pop(4))
            onnx_inp_names.insert(4, onnx_inp_names.pop(5))
            onnx_inp_names.insert(6, onnx_inp_names.pop(7))

        if "yolov8" in self.model_name:
            onnx_inp_names.insert(2, onnx_inp_names.pop(6))
            onnx_inp_names.insert(5, onnx_inp_names.pop(7))

        input_feed = {name: fmap for name, fmap in zip(onnx_inp_names, onnx_inps)}

        # Pass fmaps through onnxruntime
        onnx_out = self.ort.run(None, input_feed)
        out = torch.from_numpy(onnx_out[0])  # (B, 56, 8400)

        preds = [out, None]
        return preds
