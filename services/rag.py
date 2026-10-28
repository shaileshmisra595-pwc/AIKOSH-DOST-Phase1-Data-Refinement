from pathlib import Path
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb

class ChromaRAG:
    def __init__(self, root, run_id):
        self.root = Path(root) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.root / "chroma"))
        self.collection = self.client.get_or_create_collection(name="aikosh_chunks", metadata={"hnsw:space": "cosine"})
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
        self.docs, self.ids, self.matrix = [], [], None

    def index(self, chunks):
        self.docs = chunks
        self.ids = [f"chunk-{i+1:04d}" for i in range(len(chunks))]
        if not chunks:
            return {"model": "tfidf-local", "method": "TF-IDF unigram/bigram vectors", "dimensions": 0, "preview": []}
        self.matrix = self.vectorizer.fit_transform(chunks).toarray().astype(float)
        self.collection.upsert(ids=self.ids, documents=chunks, embeddings=self.matrix.tolist(), metadatas=[{"chunk_id": i} for i in self.ids])
        with open(self.root / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        preview = []
        for i, vector in enumerate(self.matrix[:20]):
            preview.append({"chunk_id": self.ids[i], "dimensions": int(len(vector)), "norm": round(float(np.linalg.norm(vector)), 6), "first_12_dimensions": [round(float(x), 6) for x in vector[:12]]})
        return {"model": "tfidf-local", "method": "TF-IDF unigram/bigram vectors", "dimensions": int(self.matrix.shape[1]), "preview": preview}

    def retrieve(self, query, k=5):
        if not self.docs:
            return []
        q = self.vectorizer.transform([query]).toarray().astype(float)[0]
        n = min(k, len(self.ids))
        result = self.collection.query(query_embeddings=[q.tolist()], n_results=n, include=["documents", "distances", "metadatas"])
        docs = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        return [{"chunk_id": ids[i], "distance": distances[i], "text": docs[i]} for i in range(len(docs))]
