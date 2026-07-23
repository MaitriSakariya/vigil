from ultralytics import YOLO
from pathlib import Path

# Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Dataset folder
image_dir = Path("Datasets/images")

# Run inference on every image
for image in image_dir.iterdir():
    if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        print(f"\nProcessing: {image.name}")

        results = model(image)

        for r in results:
            names = r.names
            boxes = r.boxes

            detected = []

            if boxes is not None:
                for cls in boxes.cls.tolist():
                    detected.append(names[int(cls)])

            if detected:
                print("Detected:", ", ".join(sorted(set(detected))))
            else:
                print("No objects detected.")

        # Save annotated image
        results[0].save(filename=f"Datasets/images/result_{image.name}")