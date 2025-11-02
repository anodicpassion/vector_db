# =========================================================
#  local_feature_extraction_to_chroma.py
# =========================================================
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

import chromadb
from chromadb.config import Settings

# ------------------- CONFIG -------------------
DATA_DIR = Path("Abstract_Chevron_Print_Kimono")  # <-- put your images here
CHROMA_PATH = Path("./chroma_db")  # <-- where the DB lives
COLLECTION = "fashion"  # name inside Chroma
MODEL_NAME = "resnet50"  # only choice for zero-deps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
print(f"Using device: {DEVICE}")


# ------------------- 1. FEATURE EXTRACTOR -------------------
class ResNetExtractor:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)

        # Load ResNet-50, drop the final classification layer
        backbone = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
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
        """Input: OpenCV BGR uint8 image → returns L2-normalized np.array"""
        tensor = self.transform(bgr_img).unsqueeze(0).to(self.device)
        feat = self.model(tensor)  # (1, 2048, 1, 1)
        vec = feat.squeeze().cpu().numpy()  # (2048,)
        vec = vec / np.linalg.norm(vec)  # L2 → cosine ready
        return vec


# ------------------- 2. CHROMA SETUP -------------------
def get_or_create_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False)
    )
    coll = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}  # cosine distance = dot-product of L2 vectors
    )
    return coll


# ------------------- 3. MAIN PIPELINE -------------------
def main():
    extractor = ResNetExtractor()
    collection = get_or_create_collection()

    # Gather every image
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in exts]
    print(f"Found {len(image_paths)} images → embedding…")

    for img_path in tqdm(image_paths, desc="Embedding"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        vec = extractor(img)  # (2048,)

        # Unique ID = safe filename (no path separators)
        uid = str(img_path).replace(os.sep, "_")

        meta = {
            "path": str(img_path),
            "category": img_path.parent.name,  # e.g. "tshirts"
            "filename": img_path.name,
        }

        # Upsert (idempotent)
        collection.add(
            ids=[uid],
            embeddings=[vec.tolist()],
            metadatas=[meta],
        )

    print(f"Done! All vectors are stored in {CHROMA_PATH}")


# ------------------- 4. QUICK SEARCH EXAMPLE -------------------
def demo_search(query_image_path):
    """Run after the DB is built to see nearest neighbors."""
    extractor = ResNetExtractor()
    collection = get_or_create_collection()

    q_img = cv2.imread(query_image_path)
    q_vec = extractor(q_img)

    results = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=5,
        include=["metadatas", "distances"]
    )

    print("\nTop-5 matches:")
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        print(f"  • {meta['path']}  (dist={dist:.4f})")


# ------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "search":
        # python script.py search path/to/query.jpg
        demo_search(sys.argv[2])
    else:
        main()
