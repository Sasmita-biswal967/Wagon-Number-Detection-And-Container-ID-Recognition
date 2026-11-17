
from pathlib import Path
from ultralytics import YOLO


class WNIRDetector:

    def __init__(self, model_dir="models"):

        model_dir = Path(model_dir)

        wagon_path = model_dir / "wagon_detector" / "best.pt"
        container_path = model_dir / "container_detector" / "best.pt"

        if not wagon_path.exists():
            raise FileNotFoundError(f"Missing model: {wagon_path}")

        if not container_path.exists():
            raise FileNotFoundError(f"Missing model: {container_path}")

        self.wagon_model = YOLO(str(wagon_path))
        self.container_model = YOLO(str(container_path))

    def detect(self, image, confidence=0.25, imgsz=960):

        detections = []

        models = [
            ("Wagon Number", self.wagon_model),
            ("Container Code", self.container_model)
        ]

        height, width = image.shape[:2]

        for category, model in models:

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=imgsz,
                verbose=False
            )

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].cpu().numpy()
                    )

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    detections.append({
                        "category": category,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(box.conf[0])
                    })

        return detections
