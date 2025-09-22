"""
Deployment utilities for Malayalam OCR model
"""

import os
import json
import numpy as np
from typing import Dict, List, Any

def create_model_metadata(model_path: str, labels_csv_path: str, output_path: str):
    """
    Create metadata file for model deployment
    
    Args:
        model_path: Path to trained model
        labels_csv_path: Path to labels CSV
        output_path: Where to save metadata JSON
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    # Load labels to get class information
    df = pd.read_csv(labels_csv_path)
    
    le_consonant = LabelEncoder()
    le_vowel = LabelEncoder()
    le_consonant.fit(df["consonant"])
    le_vowel.fit(df["vowel"])
    
    metadata = {
        "model_info": {
            "model_path": os.path.basename(model_path),
            "input_shape": [128, 128, 1],
            "input_dtype": "float32",
            "preprocessing": "normalize_0_1",
            "created_date": "2025-09-22"
        },
        "classes": {
            "consonant_classes": le_consonant.classes_.tolist(),
            "vowel_classes": le_vowel.classes_.tolist(),
            "num_consonant_classes": len(le_consonant.classes_),
            "num_vowel_classes": len(le_vowel.classes_)
        },
        "outputs": {
            "consonant_head": "cons",
            "vowel_head": "vow"
        },
        "usage": {
            "image_size": 128,
            "channels": 1,
            "normalization": "divide_by_255"
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Metadata saved to {output_path}")

def convert_to_tflite(model_path: str, output_path: str, quantize: bool = True):
    """
    Convert Keras model to TensorFlow Lite for mobile deployment
    
    Args:
        model_path: Path to Keras model
        output_path: Output path for .tflite file
        quantize: Whether to apply quantization for smaller size
    """
    import tensorflow as tf
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TensorFlow Lite model saved to {output_path}")
    print(f"   Size: {len(tflite_model) / 1024:.1f} KB")

def create_requirements_txt(output_path: str):
    """Create requirements.txt for deployment"""
    requirements = [
        "tensorflow>=2.12.0",
        "numpy>=1.20.0",
        "pandas>=1.2.0",
        "pillow>=8.0.0",
        "scikit-learn>=1.0.0"
    ]
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(requirements) + '\n')
    
    print(f"✅ Requirements saved to {output_path}")

def create_flask_api_example(output_path: str):
    """Create a Flask API example for web deployment"""
    flask_code = '''"""
Flask API for Malayalam OCR Model
"""

from flask import Flask, request, jsonify
import base64
import io
import numpy as np
from PIL import Image
from model_inference import MalayalamOCRModel

app = Flask(__name__)

# Initialize model (adjust paths as needed)
MODEL_PATH = "models/malayalam_ocr.keras"
LABELS_CSV_PATH = "data/labels.csv"

try:
    ocr_model = MalayalamOCRModel(MODEL_PATH, LABELS_CSV_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    ocr_model = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": ocr_model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Malayalam character from image
    
    Expected input:
    {
        "image": "base64_encoded_image_data",
        "include_probabilities": false
    }
    """
    if ocr_model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Predict
        include_probs = data.get('include_probabilities', False)
        result = ocr_model.predict(img_array, return_probabilities=include_probs)
        
        return jsonify({
            "success": True,
            "prediction": result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple images at once
    
    Expected input:
    {
        "images": ["base64_1", "base64_2", ...],
        "include_probabilities": false
    }
    """
    if ocr_model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if 'images' not in data:
            return jsonify({"error": "No images provided"}), 400
        
        images = []
        for img_b64 in data['images']:
            image_data = base64.b64decode(img_b64)
            image = Image.open(io.BytesIO(image_data))
            images.append(np.array(image))
        
        include_probs = data.get('include_probabilities', False)
        results = ocr_model.predict_batch(images, return_probabilities=include_probs)
        
        return jsonify({
            "success": True,
            "predictions": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
    
    with open(output_path, 'w') as f:
        f.write(flask_code)
    
    print(f"✅ Flask API example saved to {output_path}")

def create_deployment_package():
    """Create a complete deployment package"""
    print("🚀 Creating deployment package...")
    
    # Create directories
    os.makedirs("deployment", exist_ok=True)
    os.makedirs("deployment/models", exist_ok=True)
    os.makedirs("deployment/data", exist_ok=True)
    os.makedirs("deployment/src", exist_ok=True)
    
    # Create metadata
    best_model = "malayalam_dataset/best_model_25_0.0011.keras"
    if os.path.exists(best_model):
        create_model_metadata(
            best_model,
            "malayalam_dataset/labels.csv",
            "deployment/model_metadata.json"
        )
    
    # Create requirements
    create_requirements_txt("deployment/requirements.txt")
    
    # Create Flask API example
    create_flask_api_example("deployment/src/flask_api.py")
    
    # Create README
    readme_content = '''# Malayalam OCR Model Deployment

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Copy your trained model:**
   ```bash
   cp path/to/your/best_model_XX_X.XXXX.keras models/malayalam_ocr.keras
   cp path/to/your/labels.csv data/labels.csv
   ```

3. **Test inference:**
   ```bash
   python src/model_inference.py
   ```

4. **Run Flask API (optional):**
   ```bash
   python src/flask_api.py
   ```

## Files

- `models/`: Place your trained .keras model here
- `data/`: Place your labels.csv here  
- `src/model_inference.py`: Main inference script
- `src/flask_api.py`: Example web API
- `model_metadata.json`: Model information
- `requirements.txt`: Python dependencies

## Usage

```python
from src.model_inference import MalayalamOCRModel

# Load model
ocr = MalayalamOCRModel("models/malayalam_ocr.keras", "data/labels.csv")

# Predict single image
result = ocr.predict("image.png")
print(f"Prediction: {result['combined']}")
```

## API Usage

POST to `/predict` with base64 encoded image:
```json
{
  "image": "base64_encoded_image_data",
  "include_probabilities": false
}
```
'''
    
    with open("deployment/README.md", 'w') as f:
        f.write(readme_content)
    
    print("✅ Deployment package created in 'deployment/' directory")
    print("\nNext steps:")
    print("1. Copy your best model to deployment/models/malayalam_ocr.keras")
    print("2. Copy labels.csv to deployment/data/labels.csv")
    print("3. Follow the README instructions")

if __name__ == "__main__":
    create_deployment_package()