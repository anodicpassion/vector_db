import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


class SimpleVectorDB:
    def __init__(self, dim: int):
        """Initialize vector database with specified embedding dimension."""
        self.dim = dim
        self.vectors = []
        self.metadata = []

    def add_vector(self, vector: np.ndarray, meta: Dict[str, Any]) -> None:
        """Add a vector with associated metadata to the database."""
        if vector.shape != (self.dim,):
            raise ValueError(f"Vector dimension must be {self.dim}, got {vector.shape}")
        self.vectors.append(vector)
        self.metadata.append(meta)

    def add_batch(self, vectors: List[np.ndarray], metas: List[Dict[str, Any]]) -> None:
        """Add multiple vectors and their metadata in a batch."""
        if len(vectors) != len(metas):
            raise ValueError("Number of vectors must match number of metadata entries")
        for vec, meta in zip(vectors, metas):
            self.add_vector(vec, meta)

    def cosine_similarity(self, query: np.ndarray) -> List[Tuple[float, Dict[str, Any]]]:
        """Find similar vectors using cosine similarity, return sorted (score, metadata) pairs."""
        if not self.vectors:
            return []

        query = query.reshape(-1)
        if query.shape != (self.dim,):
            raise ValueError(f"Query dimension must be {self.dim}, got {query.shape}")

        vectors = np.array(self.vectors)
        dot_products = np.dot(vectors, query)
        vector_norms = np.linalg.norm(vectors, axis=1)
        query_norm = np.linalg.norm(query)
        similarities = dot_products / (vector_norms * query_norm + 1e-10)
        results = [(sim, self.metadata[i]) for i, sim in enumerate(similarities)]
        return sorted(results, key=lambda x: x[0], reverse=True)

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Return top_k most similar vectors with their metadata."""
        return self.cosine_similarity(query)[:top_k]


# Demo with real LLM embeddings
if __name__ == "__main__":
    # Initialize SentenceTransformers model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    db = SimpleVectorDB(dim=embedding_dim)

    # Sample data: texts to embed
    sample_texts = [
        "I love to code in Python",
        "Machine learning is fun",
        "Python is great for AI",
        "Data science is exciting"
    ]

    # Generate embeddings and add to database
    embeddings = model.encode(sample_texts)
    for text, embedding in zip(sample_texts, embeddings):
        db.add_vector(embedding, {"text": text})

    # Add more embeddings
    new_texts = [
        "AI is transforming the world",
        "Deep learning is powerful",
        "Coding is my passion"
    ]
    new_embeddings = model.encode(new_texts)
    db.add_batch(new_embeddings, [{"text": text} for text in new_texts])

    # Query with a new text
    query_text = "I enjoy programming and AI"
    query_embedding = model.encode([query_text])[0]
    results = db.search(query_embedding, top_k=5)

    # Print results
    print("Query results:")
    for score, meta in results:
        print(f"Score: {score:.3f}, Text: {meta['text']}")