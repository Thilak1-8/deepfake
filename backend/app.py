import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np
from PIL import Image

import sys
sys.path.append('..') # allow importing from parent dir
from deepfake_model import AdvancedTransform, AdvancedResNet

app = Flask(__name__, static_folder='exports')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
EXPORT_FOLDER = 'exports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPORT_FOLDER'] = EXPORT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Flask using device: {device}")

# Initialize Model & Transform
classes = ['Fake', 'Real']
print("Loading model...")
model = AdvancedResNet(num_classes=len(classes), pretrained=False)
model_path = os.path.join('..', 'best_model_advanced.pth')
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

transform = AdvancedTransform(img_size=224)
print("Model loaded successfully.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_intermediate_image(numpy_array, filename, is_colormap=False):
    """
    Saves a normalized numpy array as an image.
    If is_colormap is True, applies a nice colormap (useful for FFT/DCT).
    """
    export_path = os.path.join(app.config['EXPORT_FOLDER'], filename)
    
    # Normalize array to 0-255 uint8 range
    if numpy_array.max() == numpy_array.min():
        normalized = np.zeros_like(numpy_array, dtype=np.uint8)
    else:
        normalized = ((numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min()) * 255).astype(np.uint8)

    if is_colormap:
        # Apply Jet colormap to highlights frequencies beautifully
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        cv2.imwrite(export_path, colored)
    else:
        # Save as grayscale
        cv2.imwrite(export_path, normalized)
        
    return f"/exports/{filename}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Load Image
            image = Image.open(filepath).convert('RGB')
            
            # 2. Extract Features & Get Tensor
            input_tensor, features = transform(image, return_features=True)
            
            # 3. Predict
            input_batch = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, predicted_idx = torch.max(outputs, 1)
                predicted_idx = predicted_idx.item()
                
            predicted_class = classes[predicted_idx]
            confidence = probabilities[predicted_idx].item()
            
            # 4. Save Intermediate Visualizations
            base_name = os.path.splitext(filename)[0]
            
            # Original resized
            orig_path = os.path.join(app.config['EXPORT_FOLDER'], f"{base_name}_orig.jpg")
            orig_rgb = cv2.cvtColor(features['original'], cv2.COLOR_RGB2BGR) # Convert back to BGR for cv2
            cv2.imwrite(orig_path, orig_rgb)
            url_orig = f"/exports/{base_name}_orig.jpg"
            
            # Y-Channel (Luminance)
            url_y = save_intermediate_image(features['y_channel'], f"{base_name}_y.jpg")
            
            # FFT (Frequency Domain)
            url_fft = save_intermediate_image(features['fft'], f"{base_name}_fft.jpg", is_colormap=True)
            
            # DCT (Discrete Cosine Transform)
            url_dct = save_intermediate_image(features['dct'], f"{base_name}_dct.jpg", is_colormap=True)
            
            # Wavelet
            url_wavelet = save_intermediate_image(features['wavelet'], f"{base_name}_wavelet.jpg")
            
            return jsonify({
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    'Fake': probabilities[0].item(),
                    'Real': probabilities[1].item()
                },
                'visualizations': {
                    'original': url_orig,
                    'y_channel': url_y,
                    'fft': url_fft,
                    'dct': url_dct,
                    'wavelet': url_wavelet
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

# Route to serve exported images
@app.route('/exports/<path:filename>')
def serve_export(filename):
    return app.send_static_file(filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
