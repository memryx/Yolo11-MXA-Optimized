import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Train YOLOv11-ADHD models")
parser.add_argument('--task', type=str, choices=['detect', 'seg', 'pose'], required=True, help="Task type")
parser.add_argument('--size', type=str, choices=['n', 's', 'm'], required=True, help="Model size")
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")
parser.add_argument('--resolution', type=int, default=640, help="Input image resolution")

args = parser.parse_args()

print(f"\nTraining YOLOv11{args.size} for {args.task} task for {args.epochs} epochs at {args.resolution}x{args.resolution} resolution\n")

model = YOLO(f"./yolo11{args.size}{'-' + args.task if args.task != 'detect' else ''}-adhd.yaml")
model.train(
    epochs=args.epochs,
    imgsz=args.resolution,
    pretrained=False,
    data="coco.yaml" if args.task != 'pose' else "coco-pose.yaml",
    batch=0.999, # % of GPU memory to use
    name=f"{args.size}-{args.task}-adhd-{args.epochs}-{args.resolution}",
    project=f"runs/{args.task}"
    # See https://docs.ultralytics.com/modes/train/#train-settings for more
)
