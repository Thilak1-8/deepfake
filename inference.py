import torch
from PIL import Image
from deepfake_model import AdvancedTransform, AdvancedResNet

def predict_image(image_path, model_weights_path="best_model_advanced.pth", classes=['Fake', 'Real'], device=None):
    """
    Predicts if a given image is a Real or Fake face.
    
    Args:
        image_path (str): Path to the image file.
        model_weights_path (str): Path to the trained weights file (.pth).
        classes (list): Class names order (must match training order).
        device (str): Device to run inference on (e.g. 'cpu' or 'cuda').
        
    Returns:
        dict: Containing 'prediction', 'confidence', and 'probabilities'
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize the model
    # Note: pretrained=False since we are going to load our own trained weights anyway
    model = AdvancedResNet(num_classes=len(classes), pretrained=False)
    
    # Load the trained weights
    # We use map_location=device to allow loading a CUDA model onto CPU if needed
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 2. Initialize the transformation pipeline
    transform = AdvancedTransform(img_size=224)
    
    # 3. Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformation to get our 6-channel input
    input_tensor = transform(image)
    
    # Add batch dimension (C, H, W) -> (1, C, H, W)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # 4. Perform Inference
    with torch.no_grad():
        outputs = model(input_batch)
        
        # Calculate probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get the predicted class index
        _, predicted_idx = torch.max(outputs, 1)
        predicted_idx = predicted_idx.item()
        
    # Get the predicted class name and confidence score
    predicted_class = classes[predicted_idx]
    confidence = probabilities[predicted_idx].item()
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': {classes[i]: probabilities[i].item() for i in range(len(classes))}
    }

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Predict if a face is Real or Fake using the trained AdvancedResNet.")
    parser.add_argument("image", help="Path to the input image file")
    parser.add_argument("--weights", default="best_model_advanced.pth", help="Path to best_model_advanced.pth")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
    elif not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found.")
    else:
        print(f"Predicting on: {args.image}")
        result = predict_image(args.image, args.weights)
        print("\nResults:")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence'] * 100:.2f}%")
        print("Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob * 100:.2f}%")
