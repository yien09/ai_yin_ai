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
            
            img_h, img_w, _ = img.shape
            person_detected = False
            max_conf = 0.0
            is_centered = False
            is_size_valid = False

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
                            
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_w = x2 - x1
                        box_h = y2 - y1
                        
                        # Calculate centers
                        x_center = (x1 + x2) / 2.0
                        y_center = (y1 + y2) / 2.0
                        
                        # Check if person is centered (within 25% tolerance from the image center)
                        center_x_diff = abs(x_center - (img_w / 2.0)) / img_w
                        center_y_diff = abs(y_center - (img_h / 2.0)) / img_h
                        if center_x_diff <= 0.25 and center_y_diff <= 0.25:
                            is_centered = True
                            
                        # Check if person size is valid (occupies at least 30% of width or height)
                        if (box_w / img_w) >= 0.30 or (box_h / img_h) >= 0.30:
                            is_size_valid = True

            if person_detected:
                if not is_centered:
                    return {
                        "face_detected": False,
                        "confidence_score": max_conf,
                        "message": "Posisikan wajah Anda di tengah bingkai outline"
                    }
                if not is_size_valid:
                    return {
                        "face_detected": False,
                        "confidence_score": max_conf,
                        "message": "Dekatkan wajah Anda ke kamera sesuai bingkai"
                    }
                return {
                    "face_detected": True,
                    "confidence_score": max_conf,
                    "message": "Validasi wajah menggunakan YOLOv8 berhasil"
                }

            return {
                "face_detected": False,
                "confidence_score": 0.0,
                "message": "Wajah tidak terdeteksi oleh AI. Silakan posisikan wajah Anda ke kamera."
            }

        except Exception as e:
            logger.error(f"Error during face validation: {e}")
            return {"face_detected": False, "confidence_score": 0.0, "error": str(e)}

yolo_service = YoloService()
