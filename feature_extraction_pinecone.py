# =========================================================
#  feature_extraction_pinecone.py
# =========================================================
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

from pinecone import Pinecone, ServerlessSpec  # <-- NEW SDK
from dotenv import load_dotenv

# ------------------- LOAD ENV -------------------
load_dotenv()

# ------------------- CONFIG -------------------
DATA_DIR = Path("Abstract_Chevron_Print_Kimono")
INDEX_NAME = "fashion-index"
DIMENSION = 2048
METRIC = "cosine"
CLOUD = "aws"           # or "gcp", "azure"
REGION = "us-east-1"    # change based on your account
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- PINECONE SETUP -------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

print(f"Using device: {DEVICE}")
print(f"Pinecone index: {INDEX_NAME} ({CLOUD}/{REGION})")


# ------------------- 1. FEATURE EXTRACTOR -------------------
class ResNetExtractor:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        backbone = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, bgr_img):
        tensor = self.transform(bgr_img).unsqueeze(0).to(self.device)
        feat = self.model(tensor)
        vec = feat.squeeze().cpu().numpy()
        vec = vec / np.linalg.norm(vec)
        return vec


# ------------------- 2. INDEX SETUP -------------------
def get_or_create_index():
    # List existing indexes
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating serverless index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud=CLOUD,
                region=REGION
            )
        )
        print("Index created. Waiting for readiness...")
        # Wait until ready
        import time
        while True:
            status = pc.describe_index(INDEX_NAME).status
            if status["ready"]:
                print("Index is ready!")
                break
            print("Index not ready yet, waiting 5s...")
            time.sleep(5)
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    # Connect to index
    index = pc.Index(INDEX_NAME)
    print(f"Index stats: {index.describe_index_stats()}")
    return index


# ------------------- 3. MAIN PIPELINE -------------------
def main():
    extractor = ResNetExtractor()
    index = get_or_create_index()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in exts]
    print(f"Found {len(image_paths)} images → embedding & upserting...")

    batch_size = 32
    batch = []

    for img_path in tqdm(image_paths, desc="Processing"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        vec = extractor(img)
        uid = str(img_path).replace(os.sep, "_")
        meta = {
            "path": str(img_path),
            "category": img_path.parent.name,
            "filename": img_path.name,
        }
        batch.append((uid, vec.tolist(), meta))

        if len(batch) >= batch_size:
            index.upsert(vectors=batch)
            batch = []

    if batch:
        index.upsert(vectors=batch)

    print(f"Done! All vectors upserted to '{INDEX_NAME}'")


# ------------------- 4. SEARCH DEMO -------------------
def demo_search(query_image_path):
    extractor = ResNetExtractor()
    index = pc.Index(INDEX_NAME)

    q_img = cv2.imread(query_image_path)
    if q_img is None:
        raise FileNotFoundError(f"Query image not found: {query_image_path}")

    q_vec = extractor(q_img)

    results = index.query(
        vector=q_vec.tolist(),
        top_k=5,
        include_metadata=True
    )

    print("\nTop-5 matches:")
    for match in results["matches"]:
        meta = match["metadata"]
        print(f"  • {meta['path']}  (score={match['score']:.4f})")


# ------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "search":
        if len(sys.argv) < 3:
            print("Usage: python feature_extraction_pinecone.py search path/to/query.jpg")
            sys.exit(1)
        demo_search(sys.argv[2])
    else:
        main()