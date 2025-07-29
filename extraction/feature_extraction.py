
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path

device = torch.device("cpu")

# Load trained EfficientNet-V2-S model
model = models.efficientnet_v2_s(weights=None)
# Replicate the classifier structure used during training
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),  # Default dropout for EfficientNet-V2-S
    nn.Linear(model.classifier[1].in_features, 10)  # 10 classes
)
model.load_state_dict(torch.load("efficientnetv2s_epoch5.pth", map_location=torch.device('cpu')))  
model = nn.Sequential(*list(model.children())[:-1])  # Remove classifier for feature extraction
model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract features from a single image
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)  
        with torch.no_grad():
            features = model(img)
        return features.flatten().cpu().numpy()  # Return flattened feature vector (1280-dim for V2-S)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Main function for feature extraction
def main():
    main_folder = "data/train"  
    output_dir = "features_output"  # Directory to save features
    Path(output_dir).mkdir(exist_ok=True)

    features_list = []
    labels_list = []
    class_names = sorted(os.listdir(main_folder))  # Get class names
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    # Iterate through each class folder
    for class_name in class_names:
        class_path = os.path.join(main_folder, class_name)
        if not os.path.isdir(class_path):
            continue
        print(f"Processing class: {class_name}")
        
        # Iterate through images in the class folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            features = extract_features(img_path)
            if features is not None:
                features_list.append(features)
                labels_list.append(class_to_idx[class_name])

    # Save features and labels as NumPy arrays
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    np.save(os.path.join(output_dir, 'features.npy'), features_array)
    np.save(os.path.join(output_dir, 'labels.npy'), labels_array)
    np.save(os.path.join(output_dir, 'class_names.npy'), np.array(class_names))
    print(f"Features saved to {output_dir}/features.npy")
    print(f"Labels saved to {output_dir}/labels.npy")
    print(f"Class names saved to {output_dir}/class_names.npy")

if __name__ == "__main__":
    main()
