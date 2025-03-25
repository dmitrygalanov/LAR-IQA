import argparse
import os
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Import inference functions from your inference.py file
from inference import load_model, preprocess_image, infer

app = FastAPI()

# Global variables to hold the model and configuration parameters
model = None
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
SERVER_COLOR_SPACE = "RGB"  # default; will be overwritten by startup args if provided
MODEL_PATH = None
USE_KAN = None

@app.on_event("startup")
async def startup_event():
    global model, MODEL_PATH, USE_KAN, SERVER_COLOR_SPACE
    if MODEL_PATH is None:
        raise ValueError("MODEL_PATH must be provided at server startup")
    # Load the model once at startup using the server arguments
    model = load_model(MODEL_PATH, USE_KAN, device)
    print(f"Model loaded from {MODEL_PATH} using {'MobileNetMergedWithKAN' if USE_KAN else 'MobileNetMerged'} on {device}")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Save the uploaded image temporarily so our existing functions can use a file path
    temp_filename = f"temp_{image.filename}"
    try:
        with open(temp_filename, "wb") as f:
            content = await image.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save uploaded image.")

    try:
        # Preprocess the image using the server's default color space
        image_authentic, image_synthetic = preprocess_image(temp_filename, SERVER_COLOR_SPACE, device)
        # Run inference using the pre-loaded model
        score = infer(model, image_authentic, image_synthetic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        # Remove the temporary image file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    return JSONResponse(content={"predicted_quality_score": score})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Server for Image Quality Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--use_kan", action="store_true", help="Use MobileNetMergedWithKAN model")
    parser.add_argument("--color_space", type=str, choices=["RGB", "HSV", "LAB", "YUV"], default="RGB", help="Color space to use for inference")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    # Set globals for use during startup
    MODEL_PATH = args.model_path
    USE_KAN = args.use_kan
    SERVER_COLOR_SPACE = args.color_space

    # Run the server; note the module name ("main:app") should match your filename and app variable
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
