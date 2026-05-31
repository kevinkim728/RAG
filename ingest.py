import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

class Chunk(BaseModel):
    page_content: str
    metadata: dict


def load_documents(base_path="transcripts"):
    """
    Loops through the folders from the base_path and reads the txt files inside them
    Reads each transcript and stores text + metadata in a list of dicts
    Returns one dict(documents) per file with text, week, day, source
    """
    documents = []
    base = Path(base_path)

    for week_folder in sorted(base.iterdir()):
        if not week_folder.is_dir():
            continue
        for file in sorted(week_folder.glob("*.txt")):
            text = file.read_text(encoding="utf-8")
            documents.append({
                "text": text,
                "week": week_folder.name,
                "day": file.stem,
                "source": str(file)
            })
            print(f"Loaded: {file} ({len(text)} characters)")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Creates chunks and converts them into pydantic objects with metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["--- Lecture", "\n\n", "\n", " "]
    )

    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for piece in pieces:
            chunks.append(Chunk(
                page_content=piece,
                metadata={"week": doc["week"], "day": doc["day"], "source": doc["source"]}
            ))

    print(f"Total chunks created: {len(chunks)}")
    print(f"\n--- Sample chunk ---\n")
    print(chunks[0].page_content)
    return chunks


def embed_and_store(chunks, embedder, collection):
    """
    Create page_content, metadata, and ids and stores it in variables
    Encodes the texts into vectors
    Adds the vector into the chroma database
    """
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"{chunk.metadata['source']}_{i}" for i, chunk in enumerate(chunks)]

    print("Embedding chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Stored {len(chunks)} chunks in Chroma.")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    print(f"\nEmbedding model: {model_name}")
    embedder = SentenceTransformer(model_name, trust_remote_code=True)

    chroma = PersistentClient(path="./chroma_db")
    try:
        chroma.delete_collection("transcripts")
        print("Deleted existing collection.")
    except Exception:
        pass
    collection = chroma.get_or_create_collection("transcripts")

    documents = load_documents()
    chunks = chunk_documents(documents)
    embed_and_store(chunks, embedder, collection)

    print(f"\nDone. Update BI_ENCODER in answer.py to: '{model_name}'")
