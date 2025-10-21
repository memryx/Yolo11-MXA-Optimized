from ultralytics import YOLO
from validators import MxaDetectionValidator, MxaPoseValidator, MxaSegmentationValidator

weights = "./runs/detect/n-adhd-300-640/weights/best.pt"
task = "det"
imgsz = 640
data = "coco.yaml" if task != "pose" else "coco-pose.yaml"
validator = {
    "det": MxaDetectionValidator,
    "seg": MxaSegmentationValidator,
    "pose": MxaPoseValidator,
}[task]

model = YOLO(weights)

# Validate on CPU/CUDA
# model.val(
#     imgsz=imgsz,
#     data=data
# )

# Validate on MXA
# Export and Compile model to same directory as .pt checkpoint before running below
metrics = model.val(
    validator=validator,
    data=data,
    imgsz=imgsz,
    batch=1,
    workers=0,
    rect=False,
)
