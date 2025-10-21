from ultralytics import YOLO

task = 'det'        # 'det', 'seg', or 'pose'
epochs = 2          # default
resolution = 320    # default

for size in 'n':
    print(f"\nTraining YOLOv11{size} for {task} task for {epochs} epochs at {resolution}x{resolution} resolution\n")

    model = YOLO(f"./yolo11{size}{'-' + task if task != 'det' else ''}-adhd.yaml")
    model.train(
        epochs=epochs,
        imgsz=resolution,
        pretrained=False,
        data="coco.yaml" if task != 'pose' else "coco-pose.yaml",
        batch=0.999, # % of GPU memory to use (RTX 4090)
        name=f"{size}-{task}-adhd-{epochs}-{resolution}",
        project=f"runs/{task}"
        # See https://docs.ultralytics.com/modes/train/#train-settings for more
    )
