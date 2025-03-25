import argparse
import os
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import inference functions from your inference.py file
from inference import load_model, preprocess_image, infer

class LARIQAServer:
    def __init__(self):
        """
        Initialize the FastAPI application and global variables.
        """
        self.app = FastAPI()
        
        # Add CORS middleware for better web compatibility
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Global variables to hold the model and configuration parameters
        self.model = None
        self.device = self._select_device()
        self.model_path: Optional[str] = None
        self.use_kan: Optional[bool] = None
        self.server_color_space = "RGB"  # default; will be overwritten by startup args if provided
        
        # Register routes
        self.app.post("/predict")(self.predict)
    
    def _select_device(self) -> str:
        """
        Select the most appropriate device for model inference.
        
        Returns:
            str: Selected device ('mps', 'cuda', or 'cpu')
        """
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def startup(self, model_path: str, use_kan: bool, color_space: str):
        """
        Load the model at server startup.
        
        Args:
            model_path (str): Path to the trained model
            use_kan (bool): Flag to use KAN model variant
            color_space (str): Color space for image processing
        
        Raises:
            ValueError: If model path is not provided
        """
        if model_path is None:
            raise ValueError("MODEL_PATH must be provided at server startup")
        
        # Set configuration parameters
        self.model_path = model_path
        self.use_kan = use_kan
        self.server_color_space = color_space
        
        # Load the model once at startup using the server arguments
        self.model = load_model(self.model_path, self.use_kan, self.device)
        print(f"Model loaded from {self.model_path} using "
              f"{'MobileNetMergedWithKAN' if self.use_kan else 'MobileNetMerged'} "
              f"on {self.device}")
    
    async def predict(self, image: UploadFile = File(...)) -> JSONResponse:
        """
        Predict image quality from uploaded file.
        
        Args:
            image (UploadFile): Uploaded image file
        
        Returns:
            JSONResponse: Predicted quality score
        """
        # Save the uploaded image temporarily
        temp_filename = f"temp_{image.filename}"
        try:
            with open(temp_filename, "wb") as f:
                content = await image.read()
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to save uploaded image.")
        
        try:
            # Preprocess the image using the server's default color space
            image_authentic, image_synthetic = preprocess_image(
                temp_filename, 
                self.server_color_space, 
                self.device
            )
            
            # Run inference using the pre-loaded model
            score = infer(self.model, image_authentic, image_synthetic)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
        finally:
            # Remove the temporary image file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
        return JSONResponse(content={"predicted_quality_score": score})

def main():
    """
    Main function to parse arguments and run the server.
    """
    parser = argparse.ArgumentParser(description="API Server for Image Quality Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--use_kan", action="store_true", help="Use MobileNetMergedWithKAN model")
    parser.add_argument("--color_space", type=str, 
                        choices=["RGB", "HSV", "LAB", "YUV"], 
                        default="RGB", 
                        help="Color space to use for inference")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=30000, help="Port to run the server on")
    
    args = parser.parse_args()
    
    # Create server instance
    server = LARIQAServer()
    
    # Startup the server with provided configuration
    server.startup(
        model_path=args.model_path,
        use_kan=args.use_kan,
        color_space=args.color_space
    )
    
    # Run the server
    uvicorn.run(
        app=server.app, 
        host=args.host, 
        port=args.port, 
        reload=False
    )

if __name__ == "__main__":
    main()
