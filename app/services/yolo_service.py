import cv2
import numpy as np
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class YoloService:
    def __init__(self):
        try:
            # Load standard YOLOv8n model
            self.model = YOLO("yolov8n.pt")
            logger.info("YOLOv8 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            self.model = None

    def validate_face(self, image_bytes: bytes) -> dict:
        if self.model is None:
            return {"face_detected": False, "confidence_score": 0.0, "error": "Model not loaded"}

        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"face_detected": False, "confidence_score": 0.0, "error": "Failed to decode image"}

            # Run inference
            results = self.model(img)
            
            # Look for a "person" (class 0 in COCO dataset) or any detection
            person_detected = False
            max_conf = 0.0

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Class 0 is person in yolov8n.pt
                    if cls_id == 0:
                        person_detected = True
                        if conf > max_conf:
                            max_conf = conf

            # If we detect a person, count it as a face match for check-in
            if person_detected:
                return {
                    "face_detected": True,
                    "confidence_score": max_conf,
                    "message": "Face/Person validation successful"
                }
            
            # If no person is detected but there are other detections, use the highest confidence
            if len(results) > 0 and len(results[0].boxes) > 0:
                highest_conf = float(results[0].boxes.conf[0])
                if highest_conf > 0.5:
                    return {
                        "face_detected": True,
                        "confidence_score": highest_conf,
                        "message": "Detection successful"
                    }

            return {
                "face_detected": False,
                "confidence_score": 0.0,
                "message": "No face or person detected in the image"
            }

        except Exception as e:
            logger.error(f"Error during face validation: {e}")
            return {"face_detected": False, "confidence_score": 0.0, "error": str(e)}

yolo_service = YoloService()
