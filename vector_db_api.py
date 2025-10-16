import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import uuid


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


# Initialize Flask app and vector database
app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = model.get_sentence_embedding_dimension()
db = SimpleVectorDB(dim=embedding_dim)


@app.route('/add_vector', methods=['POST'])
def add_vector():
    """API endpoint to add a single vector with text and optional metadata."""
    try:
        data = request.get_json()
        text = data.get('text')
        metadata = data.get('metadata', {})

        if not text:
            return jsonify({'error': 'Text is required'}), 400

        # Generate embedding
        embedding = model.encode([text])[0]

        # Add ID to metadata if not provided
        if 'id' not in metadata:
            metadata['id'] = str(uuid.uuid4())
        metadata['text'] = text

        # Add to database
        db.add_vector(embedding, metadata)

        return jsonify({
            'message': 'Vector added successfully',
            'metadata': metadata
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/add_batch', methods=['POST'])
def add_batch():
    """API endpoint to add multiple vectors with texts and metadata."""
    try:
        data = request.get_json()
        items = data.get('items', [])

        if not items:
            return jsonify({'error': 'Items list is required'}), 400

        vectors = []
        metas = []
        for item in items:
            text = item.get('text')
            if not text:
                return jsonify({'error': 'Each item must have text'}), 400
            metadata = item.get('metadata', {})
            metadata['text'] = text
            if 'id' not in metadata:
                metadata['id'] = str(uuid.uuid4())
            vectors.append(model.encode([text])[0])
            metas.append(metadata)

        # Add batch to database
        db.add_batch(vectors, metas)

        return jsonify({
            'message': f'Added {len(vectors)} vectors successfully',
            'metadata': metas
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """API endpoint to search for similar vectors given a query text."""
    try:
        data = request.get_json()
        query_text = data.get('query_text')
        top_k = data.get('top_k', 5)

        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400

        # Generate query embedding
        query_embedding = model.encode([query_text])[0]

        # Perform search
        results = db.search(query_embedding, top_k=top_k)

        # Format results
        formatted_results = [
            {'score': float(score), 'metadata': meta}
            for score, meta in results
        ]

        return jsonify({
            'message': 'Search completed',
            'results': formatted_results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """API endpoint to check database status."""
    return jsonify({
        'message': 'Vector database is running',
        'vector_count': len(db.vectors),
        'embedding_dim': db.dim
    }), 200


if __name__ == "__main__":
    # Add some sample data for testing
    sample_texts = [
        "I love to code in Python",
        "Machine learning is fun",
        "Python is great for AI",
        "Data science is exciting"
    ]
    embeddings = model.encode(sample_texts)
    db.add_batch(embeddings, [{"text": text, "id": str(uuid.uuid4())} for text in sample_texts])

    # Run Flask app
    app.run(debug=True)