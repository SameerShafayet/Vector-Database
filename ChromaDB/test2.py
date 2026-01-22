from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import EmbeddingFunction

# ---- Define embedding class ----
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        return self.model.encode(input).tolist()

# ---- Create embedding function ----
embedding_function = SentenceTransformerEmbeddingFunction()

# ---- Create Chroma client & collection ----
client = chromadb.Client()
collection = client.create_collection(
    name="custom_embeddings",
    embedding_function=embedding_function
)

# ---- Add documents ----
collection.add(
    documents=[
        "Vision transformers are powerful for image understanding",
        "CNNs are widely used for image classification",
        "Transformers dominate modern NLP tasks",
        "Large language models benefit from retrieval augmented generation"
    ],
    metadatas=[
        {"domain": "cv"},
        {"domain": "cv"},
        {"domain": "nlp"},
        {"domain": "llm"}
    ],
    ids=["doc1", "doc2", "doc3", "doc4"]
)

# ---- Query ----
results = collection.query(
    query_texts=["deep learning models for images"],
    n_results=2
)

# ---- Print results ----
print("Top documents:")
for doc in results["documents"][0]:
    print("-", doc)

print("\nMetadata:")
for meta in results["metadatas"][0]:
    print("-", meta)