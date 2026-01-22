import chromadb

client = chromadb.Client()
collection = client.create_collection(name="my_collection")

collection.add(
    documents=[
        "Vision transformers are powerful",
        "CNNs are good for images",
        "Transformers dominate NLP"
    ],
    metadatas=[
        {"type": "cv"},
        {"type": "cv"},
        {"type": "nlp"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

results = collection.query(
    query_texts=["deep learning for images"],  ## one query only
    n_results=2
)

print(results["documents"])

# ---- Inspect results ----
print("Top documents:")
for doc in results["documents"][0]:
    print("-", doc)

print("\nMetadata:")
for meta in results["metadatas"][0]:
    print("-", meta)
