# Malayalam OCR Model - Usage Guide

## Overview
This guide shows how to use your trained Malayalam OCR model for inference in other projects.

## What You Need to Copy

### 1. Model Files
- **Best Model**: `malayalam_dataset/best_model_25_0.0011.keras` (lowest validation loss)
- **Alternative**: `malayalam_dataset/two_headed_cnn.keras` (final model)

### 2. Label Encoders Data
- **Labels CSV**: `malayalam_dataset/labels.csv` (needed to recreate label encoders)

### 3. Model Configuration
- **Image Size**: 128x128 pixels
- **Input Format**: Grayscale (1 channel)
- **Output**: Two heads (consonant + vowel predictions)

## Usage Examples

### Basic Inference Script
See `model_inference.py` for a complete example.

### Integration Steps
1. Copy model file (.keras) to your project
2. Copy labels.csv to recreate label encoders
3. Install dependencies: `tensorflow`, `numpy`, `pandas`, `pillow`, `scikit-learn`
4. Use the inference code as shown in the example

### Model Performance
- **Consonant Accuracy**: ~99%+ 
- **Vowel Accuracy**: ~99%+
- **Combined Accuracy**: ~98%+
- **Classes**: 
  - Consonants: ~50+ classes (including vowels, consonants, chillaksharams, consonant+vowel combinations)
  - Vowels: ~13+ classes (including NONE, various vowel signs)

## Deployment Options

### 1. Python Script/Application
- Use the provided inference script directly
- Integrate into larger Python applications

### 2. Web API (Flask/FastAPI)
- Wrap the model in a REST API
- Accept image uploads, return predictions

### 3. Mobile/Edge Deployment
- Convert to TensorFlow Lite for mobile apps
- Use TensorFlow.js for web browsers

### 4. Batch Processing
- Process multiple images at once
- Save results to CSV/database

## File Structure for Deployment
```
your_project/
├── models/
│   └── malayalam_ocr.keras          # Your trained model
├── data/
│   └── labels.csv                   # For recreating encoders
├── src/
│   ├── model_inference.py           # Inference script
│   └── utils.py                     # Helper functions
├── requirements.txt                 # Dependencies
└── README.md                        # Usage instructions
```

## Dependencies
```
tensorflow>=2.12.0
numpy>=1.20.0
pandas>=1.2.0
pillow>=8.0.0
scikit-learn>=1.0.0
```