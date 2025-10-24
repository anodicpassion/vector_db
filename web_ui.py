# =========================================================
#  web_ui.py  –  Offline Fashion Visual Search with Gradio
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

import gradio as gr

# ------------------- CONFIG -------------------
CHROMA_PATH = Path("./chroma_db")
COLLECTION  = "fashion"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- 1. RESNET EXTRACTOR -------------------
class ResNetExtractor:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)

        # Use the new `weights` API (silences the warning)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(backbone.children())[:-1])   # drop FC
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, bgr_img):
        tensor = self.transform(bgr_img).unsqueeze(0).to(self.device)
        vec = self.model(tensor).squeeze().cpu().numpy()
        vec = vec / np.linalg.norm(vec)          # L2 → cosine
        return vec

# ------------------- 2. CHROMA CLIENT -------------------
def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False)
    )
    # Create if missing (idempotent)
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

# ------------------- 3. SEARCH FUNCTION -------------------
def search(image_pil, top_k, filter_category):
    """
    image_pil : PIL.Image from Gradio
    top_k     : int
    filter_category : str or None
    Returns list of image paths (strings)
    """
    # Convert PIL → OpenCV BGR
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    extractor = ResNetExtractor()
    collection = get_collection()

    q_vec = extractor(img_cv)

    where = {"category": filter_category} if filter_category else None

    results = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=top_k,
        where=where,
        include=["metadatas"]
    )

    # Return absolute paths so Gradio can display them
    paths = [m["path"] for m in results["metadatas"][0]]
    return paths

# ------------------- 4. GRADIO INTERFACE -------------------
with gr.Blocks(title="Fashion Visual Search (Offline)") as demo:
    gr.Markdown("# Fashion Visual Search – 100% Local")
    gr.Markdown("Drag an image → see the most similar items from your catalog.")

    with gr.Row():
        with gr.Column(scale=2):
            input_img = gr.Image(type="pil", label="Query Image")
        with gr.Column(scale=1):
            top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K")
            cat_filter = gr.Dropdown(
                choices=["", "tshirts", "skirts", "dresses"],  # add your own
                label="Filter by Category (optional)",
                value=""
            )

    btn = gr.Button("Search")
    gallery = gr.Gallery(label="Similar Images", columns=3, height="auto")

    btn.click(
        fn=search,
        inputs=[input_img, top_k, cat_filter],
        outputs=gallery
    )

# ------------------- 5. LAUNCH -------------------
if __name__ == "__main__":
    # Make sure the DB exists (run the ingestion script first!)
    if not CHROMA_PATH.exists():
        gr.Warning("Chroma DB not found! Run the ingestion script first.")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False   # set True only if you want a temporary public link
    )