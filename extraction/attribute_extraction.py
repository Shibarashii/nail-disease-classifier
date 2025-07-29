
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device("cpu")

# Load trained EfficientNet-V2-S model
model = models.efficientnet_v2_s(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, 10)  # 10 classes
)
model.load_state_dict(torch.load("efficientnetv2s_epoch5.pth", map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Define class names
class_names = [
    'Acral Lentiginous Melanoma', 'Beaus Line', 'Blue Finger', 'Clubbing',
    'Healthy Nail', 'Koilonychia', 'Muehrckes Lines', 'Onychogryphosis',
    'Pitting', 'Terry-s Nail'
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def save_gradient(grad):
            self.gradients = grad
        def save_activation(module, input, output):
            self.activations = output
        target_layer.register_forward_hook(save_activation)
        target_layer.register_full_backward_hook(lambda module, grad_in, grad_out: save_gradient(grad_out[0]))

    def generate(self, x, class_idx=None):
        x = x.requires_grad_()
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        score = output[0, class_idx]
        self.model.zero_grad()
        score.backward()
        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.squeeze().detach().cpu().numpy(), class_idx

# Function to create binary mask
def create_binary_mask(heatmap, threshold=0.5):
    binary_mask = (heatmap > threshold).astype(np.uint8) * 255
    return binary_mask

# Function to overlay heatmap on image
def overlay_heatmap(heatmap, image, alpha=0.5):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# Attribute descriptions
def get_attribute_description(class_name):
    descriptions = {
        'Acral Lentiginous Melanoma': 'Irregular brown-black longitudinal bands, possibly with blurred borders or pigment extending to nail fold (Hutchinson’s sign).',
        'Beaus Line': 'Transverse grooves or depressions across the nail plate.',
        'Blue Finger': 'Blue or cyanotic discoloration of the nail bed.',
        'Clubbing': 'Bulbous enlargement of distal digit with increased nail curvature.',
        'Healthy Nail': 'Uniform pink nail bed, smooth surface, no discoloration or deformities.',
        'Koilonychia': 'Spoon-shaped, concave nail surface.',
        'Muehrckes Lines': 'Paired white transverse bands parallel to the lunula.',
        'Onychogryphosis': 'Thickened, curved, claw-like nails.',
        'Pitting': 'Small, punctate depressions in the nail plate.',
        'Terry-s Nail': 'White nail bed with a narrow (1-2mm) pink or brown distal band.'
    }
    return descriptions.get(class_name, 'No description available.')

# Main function to process all images
def main():
    main_folder = "data/train"
    output_dir = "attribute_output"
    Path(output_dir).mkdir(exist_ok=True)

    # Initialize Grad-CAM
    target_layer = model.features[-1]  # Last conv layer in EfficientNet-V2-S
    grad_cam = GradCAM(model, target_layer)

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(main_folder, class_name)
        if not os.path.isdir(class_path):
            print(f"Directory {class_path} not found")
            continue
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files:
            print(f"No images found in {class_path}")
            continue

        for img_name in img_files:
            img_path = os.path.join(class_path, img_name)
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                orig_img = np.array(img.resize((224, 224)))

                # Generate Grad-CAM heatmap
                heatmap, pred_idx = grad_cam.generate(img_tensor, class_idx=class_idx)
                heatmap_img = overlay_heatmap(heatmap, orig_img)
                binary_mask = create_binary_mask(heatmap)
                grayscale_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)

                # Create 4x4 plot
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                axes[0, 0].imshow(orig_img)
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis('off')
                axes[0, 1].imshow(grayscale_img, cmap='gray')
                axes[0, 1].set_title("Grayscale Image")
                axes[0, 1].axis('off')
                axes[1, 0].imshow(heatmap_img)
                axes[1, 0].set_title("Detected Attribute (Grad-CAM)")
                axes[1, 0].axis('off')
                axes[1, 1].imshow(binary_mask, cmap='gray')
                axes[1, 1].set_title("Binary Mask")
                axes[1, 1].axis('off')

                # Add text description
                description = (f"Class: {class_name}\n"
                               f"Predicted: {class_names[pred_idx]}\n"
                               f"Attribute: {get_attribute_description(class_name)}")
                plt.figtext(0.1, 0.01, description, wrap=True, fontsize=10)

                # Save plot
                output_path = os.path.join(output_dir, f"{class_name}_{img_name}_attributes.jpg")
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
                print(f"Saved plot for {class_name}/{img_name} to {output_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    main()
