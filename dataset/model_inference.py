"""
Malayalam OCR Model Inference Script

This script demonstrates how to load and use the trained Malayalam OCR model
for inference on new images.

Author: Generated for Malayalam OCR project
Date: September 2025
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MalayalamOCRModel:
    """
    Malayalam OCR model wrapper for easy inference
    """
    
    def __init__(self, model_path, labels_csv_path, img_size=128, encodings_path=None):
        """
        Initialize the Malayalam OCR model
        
        Args:
            model_path (str): Path to the trained .keras model file
            labels_csv_path (str): Path to labels.csv file used during training
            encodings_path (str|None): Optional path to encodings.json saved during training
            img_size (int): Image size expected by model (default: 128)
        """
        self.img_size = img_size
        self.model_path = model_path
        self.labels_csv_path = labels_csv_path
        self.encodings_path = encodings_path
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Recreate label encoders from training data
        self._setup_label_encoders()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Consonant classes: {len(self.le_consonant.classes_)}")
        logger.info(f"Vowel classes: {len(self.le_vowel.classes_)}")
    
    def _setup_label_encoders(self):
        """Recreate label encoders from saved encodings.json if available, else from labels.csv"""
        self.le_consonant = LabelEncoder()
        self.le_vowel = LabelEncoder()

        # Prefer explicit encodings to guarantee class order
        if self.encodings_path and os.path.exists(self.encodings_path):
            logger.info(f"Loading class encodings from {self.encodings_path}")
            try:
                with open(self.encodings_path, "r", encoding="utf-8") as f:
                    enc = json.load(f)
                cons_classes = enc.get("consonant_classes")
                vow_classes = enc.get("vowel_classes")
                if not cons_classes or not vow_classes:
                    raise ValueError("encodings.json missing required keys")
                # Set classes_ directly for decoding; also fit encoders minimally for API compat
                self.le_consonant.classes_ = np.array(cons_classes, dtype=object)
                self.le_vowel.classes_ = np.array(vow_classes, dtype=object)
                logger.info("Loaded encodings.json successfully")
            except Exception as e:
                logger.warning(f"Failed to load encodings.json ({e}); falling back to labels.csv")
                self._fit_encoders_from_csv()
        else:
            # Fallback to CSV
            self._fit_encoders_from_csv()

        # Create inverse mappings for decoding predictions
        self.inv_consonant = {i: c for i, c in enumerate(self.le_consonant.classes_)}
        self.inv_vowel = {i: v for i, v in enumerate(self.le_vowel.classes_)}

    def _fit_encoders_from_csv(self):
        if not self.labels_csv_path or not os.path.exists(self.labels_csv_path):
            raise FileNotFoundError(
                "labels_csv_path not found and encodings.json not provided; cannot build encoders"
            )
        logger.info(f"Loading labels from {self.labels_csv_path}")
        df = pd.read_csv(self.labels_csv_path)
        self.le_consonant.fit(df["consonant"])
        self.le_vowel.fit(df["vowel"])
    
    def preprocess_image(self, image_path_or_array):
        """
        Preprocess an image for model inference
        
        Args:
            image_path_or_array: Either a file path (str) or numpy array
            
        Returns:
            np.ndarray: Preprocessed image ready for model
        """
        if isinstance(image_path_or_array, str):
            # Load image from file
            img = Image.open(image_path_or_array).convert('L')  # Convert to grayscale
            img = img.resize((self.img_size, self.img_size))
            img_array = np.array(img)
        else:
            # Assume it's already a numpy array
            img_array = image_path_or_array
            if len(img_array.shape) == 3:
                # Convert RGB to grayscale if needed
                img_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Normalize to [0, 1] and add batch dimension
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
        
        return img_array
    
    def predict(self, image_path_or_array, return_probabilities=False):
        """
        Predict Malayalam character from image
        
        Args:
            image_path_or_array: Image file path or numpy array
            return_probabilities (bool): If True, return prediction probabilities
            
        Returns:
            dict: Prediction results with consonant and vowel
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path_or_array)
        
        # Get model predictions
        predictions = self.model.predict(img_array, verbose=0)
        
        # Get predicted classes
        cons_pred_idx = np.argmax(predictions["cons"][0])
        vow_pred_idx = np.argmax(predictions["vow"][0])
        
        # Decode predictions
        consonant = self.inv_consonant[cons_pred_idx]
        vowel = self.inv_vowel[vow_pred_idx]
        
        result = {
            "consonant": consonant,
            "vowel": vowel,
            "combined": f"{consonant}+{vowel}" if vowel != "NONE" else consonant
        }
        
        if return_probabilities:
            result["consonant_confidence"] = float(np.max(tf.nn.softmax(predictions["cons"][0])))
            result["vowel_confidence"] = float(np.max(tf.nn.softmax(predictions["vow"][0])))
            result["consonant_probabilities"] = tf.nn.softmax(predictions["cons"][0]).numpy()
            result["vowel_probabilities"] = tf.nn.softmax(predictions["vow"][0]).numpy()
        
        return result
    
    def predict_batch(self, image_paths_or_arrays, return_probabilities=False):
        """
        Predict multiple images at once
        
        Args:
            image_paths_or_arrays: List of image paths or arrays
            return_probabilities (bool): If True, return prediction probabilities
            
        Returns:
            list: List of prediction results
        """
        results = []
        for img in image_paths_or_arrays:
            result = self.predict(img, return_probabilities)
            results.append(result)
        return results


def main():
    """
    Example usage of the Malayalam OCR model
    """
    # Paths - adjust these for your setup
    model_path = "dataset/malayalam_dataset/best_model_17_0.0122.keras"  # Best model
    labels_csv_path = "dataset/malayalam_dataset/labels.csv"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Available models:")
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.keras'):
                    print(f"  - {f}")
        return
    
    if not os.path.exists(labels_csv_path):
        print(f"❌ Labels file not found: {labels_csv_path}")
        return
    
    # Initialize model
    try:
        ocr_model = MalayalamOCRModel(model_path, labels_csv_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with some images (if available)
    image_dir = "dataset/malayalam_dataset/images"
    if os.path.exists(image_dir):
        # Get a few sample images
        sample_images = [f for f in os.listdir(image_dir) if f.endswith('.png')][:5]
        
        print(f"\n🔍 Testing with {len(sample_images)} sample images:")
        print("-" * 60)
        
        for img_name in sample_images:
            img_path = os.path.join(image_dir, img_name)
            try:
                result = ocr_model.predict(img_path, return_probabilities=True)
                print(f"Image: {img_name}")
                print(f"  Prediction: {result['combined']}")
                print(f"  Consonant: {result['consonant']} (conf: {result['consonant_confidence']:.3f})")
                print(f"  Vowel: {result['vowel']} (conf: {result['vowel_confidence']:.3f})")
                print()
            except Exception as e:
                print(f"  ❌ Error predicting {img_name}: {e}")
    else:
        print(f"\n📁 No test images found in {image_dir}")
        print("To test with your own images:")
        print("  result = ocr_model.predict('path/to/your/image.png')")
        print("  print(result)")
    
    # Show model info
    print("\n📋 Model Information:")
    print(f"  Consonant classes: {len(ocr_model.le_consonant.classes_)}")
    print(f"  Vowel classes: {len(ocr_model.le_vowel.classes_)}")
    print(f"  Input size: {ocr_model.img_size}x{ocr_model.img_size} grayscale")

if __name__ == "__main__":
    main()