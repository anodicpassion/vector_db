import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import numpy as np

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)
model.eval()

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
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        embeddings.append(image_features.cpu().numpy())
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
output_dir = "clustered_images"
os.makedirs(output_dir, exist_ok=True)
for cluster_id, img_list in clusters.items():
    cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        os.rename(img_path, os.path.join(cluster_dir, img_name))

print(f"Clustered images saved to {output_dir}")