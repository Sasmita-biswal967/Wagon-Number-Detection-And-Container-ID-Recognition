
import cv2
import numpy as np
import easyocr


class TextRecognizer:

    def __init__(self):

        self.reader = easyocr.Reader(
            ["en"],
            gpu=self._gpu_available()
        )

    @staticmethod
    def _gpu_available():

        import torch
        return torch.cuda.is_available()

    def preprocess(self, crop):

        if crop is None or crop.size == 0:
            return None

        height, width = crop.shape[:2]

        crop = cv2.resize(
            crop,
            (width * 3, height * 3),
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(gray)

    def recognize(self, crop):

        processed = self.preprocess(crop)

        if processed is None:
            return "", 0.0

        results = self.reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        if not results:
            return "", 0.0

        results.sort(
            key=lambda item: min(point[0] for point in item[0])
        )

        text = "".join(
            item[1].upper().replace(" ", "")
            for item in results
        )

        confidence = float(
            np.mean([item[2] for item in results])
        )

        return text, confidence
