from fastapi import FastAPI, UploadFile, File, HTTPException, status
from app.services.yolo_service import yolo_service
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI.Yin AI Microservice", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": yolo_service.model is not None}

@app.post("/api/ai/validate-face")
async def validate_face(file: UploadFile = File(...)):
    # Validate content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image"
        )
    
    try:
        # Read file bytes
        image_bytes = await file.read()
        
        # Run YOLO face detection
        result = yolo_service.validate_face(image_bytes)
        
        logger.info(f"Face validation processed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error reading upload file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating face: {str(e)}"
        )
