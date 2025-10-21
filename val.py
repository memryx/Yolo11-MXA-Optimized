import argparse
from ultralytics import YOLO
from validators import MxaDetectionValidator, MxaPoseValidator, MxaSegmentationValidator

parser = argparse.ArgumentParser(description="Validate YOLOv11-ADHD models")
parser.add_argument('--weights', type=str, required=True, help="Path to model weights (.pt)")
parser.add_argument('--device', type=str, choices=['mxa', 'cuda', 'cpu'], required=True, help="Device to validate on")
parser.add_argument('--task', type=str, choices=['detect', 'seg', 'pose'], default='detect', help="Task type")
parser.add_argument('--resolution', type=int, default=640, help="Input image resolution")

args = parser.parse_args()

data = "coco.yaml" if args.task != "pose" else "coco-pose.yaml"
model = YOLO(args.weights)

if args.device == 'mxa':
    validator = {
        "detect": MxaDetectionValidator,
        "seg": MxaSegmentationValidator,
        "pose": MxaPoseValidator,
    }[args.task]
    # Export and Compile model to same directory as .pt checkpoint before running below
    metrics = model.val(
        validator=validator,
        data=data,
        imgsz=args.resolution,
        batch=1,
        workers=0,
        rect=False,
    )
else:
    # Validate on CPU/CUDA
    metrics = model.val(
        imgsz=args.resolution,
        data=data,
    )
