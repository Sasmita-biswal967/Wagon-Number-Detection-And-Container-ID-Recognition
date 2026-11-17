
import cv2
import pandas as pd

from src.detector import WNIRDetector
from src.ocr import TextRecognizer


class WNIRPipeline:

    def __init__(self, model_dir="models"):

        self.detector = WNIRDetector(model_dir)
        self.recognizer = TextRecognizer()

    def predict(self, image, confidence=0.25):

        annotated = image.copy()

        detections = self.detector.detect(
            image,
            confidence=confidence
        )

        records = []

        for detection in detections:

            category = detection["category"]
            x1, y1, x2, y2 = detection["bbox"]

            detection_confidence = detection["confidence"]

            crop = image[y1:y2, x1:x2]

            text, ocr_confidence = self.recognizer.recognize(crop)

            display_text = text if text else "Unreadable"

            color = (0, 255, 0)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"{category}: {display_text}"

            cv2.putText(
                annotated,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            records.append({
                "Category": category,
                "Recognized Text": text,
                "Detection Confidence": round(
                    detection_confidence, 4
                ),
                "OCR Confidence": round(
                    ocr_confidence, 4
                ),
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2
            })

        results_df = pd.DataFrame(records)

        return annotated, results_df
