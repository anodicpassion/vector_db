# =========================================================
#  use_chroma.py  –  Query & explore your fashion vector DB
# =========================================================
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

import chromadb
from chromadb.config import Settings

# ------------------- CONFIG -------------------
CHROMA_PATH = Path("./chroma_db")
COLLECTION = "fashion"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------- 1. RESNET EXTRACTOR (same as before) -------------------
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
        vec = self.model(tensor).squeeze().cpu().numpy()
        vec = vec / np.linalg.norm(vec)
        return vec


# ------------------- 2. CHROMA CLIENT -------------------
def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(name=COLLECTION)


# ------------------- 3. SEARCH BY IMAGE -------------------
def search_by_image(query_path, top_k=5, filter_category=None):
    extractor = ResNetExtractor()
    collection = get_collection()

    img = cv2.imread(str(query_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {query_path}")

    q_vec = extractor(img)

    where = {"category": filter_category} if filter_category else None

    results = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=top_k,
        where=where,
        include=["metadatas", "distances", "documents"]
    )

    print(f"\nTop {top_k} similar images" + (f" in '{filter_category}'" if filter_category else "") + ":")
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        print(f"  • {meta['filename']:30} | dist={dist:.4f} | {meta['path']}")

    return results


# ------------------- 4. LIST ALL IN CATEGORY -------------------
def list_category(category):
    collection = get_collection()
    results = collection.get(
        where={"category": category},
        include=["metadatas"]
    )
    print(f"\nAll images in '{category}' ({len(results['metadatas'])}):")
    for meta in results["metadatas"]:
        print(f"  • {meta['filename']}")


# ------------------- 5. DELETE IMAGE -------------------
def delete_image(image_path):
    uid = str(image_path).replace(os.sep, "_")
    collection = get_collection()
    collection.delete(ids=[uid])
    print(f"Deleted {image_path}")


# ------------------- 6. UPDATE METADATA (e.g. change category) -------------------
def update_category(image_path, new_category):
    uid = str(image_path).replace(os.sep, "_")
    collection = get_collection()
    collection.update(
        ids=[uid],
        metadatas=[{"category": new_category}]
    )
    print(f"Updated category of {image_path} → {new_category}")


# ------------------- MAIN CLI -------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Use your fashion vector DB")
    sub = parser.add_subparsers(dest="cmd")

    # search
    s = sub.add_parser("search", help="Find similar images")
    s.add_argument("image", help="Path to query image")
    s.add_argument("--top", type=int, default=5, help="Number of results")
    s.add_argument("--cat", help="Filter by category")

    # list
    l = sub.add_parser("list", help="List all in a category")
    l.add_argument("category", help="e.g. tshirts")

    # delete
    d = sub.add_parser("delete", help="Remove an image")
    d.add_argument("image", help="Path to image to delete")

    # update
    u = sub.add_parser("update-cat", help="Change category")
    u.add_argument("image", help="Path to image")
    u.add_argument("newcat", help="New category name")

    args = parser.parse_args()

    if args.cmd == "search":
        search_by_image(args.image, top_k=args.top, filter_category=args.cat)
    elif args.cmd == "list":
        list_category(args.category)
    elif args.cmd == "delete":
        delete_image(args.image)
    elif args.cmd == "update-cat":
        update_category(args.image, args.newcat)
    else:
        parser.print_help()
