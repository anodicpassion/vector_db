import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.cluster import KMeans
import numpy as np

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet50 model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = model.to(device)
model.eval()

# Remove the final classification layer to get features
model = torch.nn.Sequential(*list(model.children())[:-1])  # Output shape: (batch_size, 2048, 1, 1)

# Image preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directory containing images
image_dir = "path/to/your/image/folder"  # Replace with your image folder path
image_extensions = (".jpg", ".jpeg", ".png")

# Load images
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
images = [Image.open(img_path).convert("RGB") for img_path in image_paths]

# Generate embeddings
def get_image_embeddings(images):
    embeddings = []
    batch_size = 32  # Adjust based on your GPU memory
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            features = model(inputs)
        features = features.view(features.size(0), -1)  # Flatten to (batch_size, 2048)
        embeddings.append(features.cpu().numpy())
    return np.vstack(embeddings)

# Get embeddings
embeddings = get_image_embeddings(images)

# Apply K-Means clustering
n_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Organize images by cluster
clusters = {i: [] for i in range(n_clusters)}
for img_path, label in zip(image_paths, cluster_labels):
    clusters[label].append(img_path)

# Print results
for cluster_id, img_list in clusters.items():
    print(f"Cluster {cluster_id}: {len(img_list)} images")
    for img_path in img_list[:5]:  # Print first 5 images per cluster
        print(f"  - {img_path}")

# Optionally, save clusters to separate folders
output_dir = "clustered_images_resnet"
os.makedirs(output_dir, exist_ok=True)
for cluster_id, img_list in clusters.items():
    cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        os.rename(img_path, os.path.join(cluster_dir, img_name))  # Moves images; use shutil.copy if you want to copy instead

print(f"Clustered images saved to {output_dir}")