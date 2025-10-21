# YOLO11 MXA Optimized

This repo contains the code used to generate the MXA-optimized versions of YOLO11. Below we outline the methodology.

## Setup 

Create a python virtual environment, install `ultralytics==8.3.152` and `memryx`.

## Architecture

YOLO11 uses the C2PSA block at the end of it's backbone. This is an Attention-based block which does not run efficiently on the MXA. We optimized here by removing this block while making all the Convolution layers in the model wider (more output channels).

The model architecture is defined in `.yaml` files. This repo includes both the default and our custom architectures. To see an exact description of our modifications, run:

```bash
diff yolo11.yaml yolo11-adhd.yaml
diff yolo11-pose.yaml yolo11-pose-adhd.yaml
diff yolo11-seg.yaml yolo11-seg-adhd.yaml
```

## Training

The `train.py` script shows how we trained. While there are many hyperparameters to tune, we found `epochs`, `resolution`, and the model architecture (`width`) had the most significant impact. Most hyperparameters were kept at their default value. All our training was done on an NVIDIA GeForce RTX4090.

We suspect some extra accuracy (~1%) can be squeezed out with further hyperparameter tuning.

## Export

```bash
yolo export model=[path to model checkpoint or yaml config] imgsz=[desired image size] format=onnx simplify=True
```

## Compilation

```bash
mx_nc -m [path to model .pt checkpoint] -e hard -j max --autocrop
```

Omit `-e hard -j max` for faster iteration. If compiling default model (with C2PSA block), also include `--extensions Yolov10`.

## Validation

The `val.py` script shows how we validated on CUDA and on our MXAs. For the latter case, we define custom validators in `validators.py`. 

Note, these validators require the MemryX SDK, the exact version of ultralytics above, and the `.pt`, the exported `.onnx`, and the compiled `.dfp` versions of the model.

Again, we have kept almost all hyperparameters default to ensure a fair comparison. The primary metric we have optimized for is $`mAP_{50:95}^{val}`$ bounding box for detection, keypoints for pose estimation, and mask for segmentation.
