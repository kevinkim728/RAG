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
    documents = [] #List of dicts
    base = Path(base_path) #Converts base_path into a Path object to use .glob and .iterdir

    # Load documents from the transcripts folder
    # If its not a folder, continue
    # If it is a folder, iterate through all files in the folder
    for week_folder in sorted(base.iterdir()):
        if not week_folder.is_dir():
            continue
        # Iterate through all files in the week folder using .glob
        for file in sorted(week_folder.glob("*.txt")):
            text = file.read_text(encoding="utf-8")
            # Create a dictionary for each document and add it to the list
            documents.append({
                "text": text, # text of the transcript
                "week": week_folder.name, # name of the week folder
                "day": file.stem, # name of the file
                "source": str(file) # path of the file. file is a Path object and its converted to a string
            })
            print(f"Loaded: {file} ({len(text)} characters)")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Creates chunks and converts them into pydantic objects with metadata
    """
    # Creates the chunks using Langchains RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["--- Lecture", "\n\n", "\n", " "]
    )

    # Loops on the documents list which is a list of dicts from the cell above
    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"]) # split_text splits documents into chunks based on the parameters from RecursiveCharacterTextSplitter.
        # Puts in pydantic structure for structured outputs and dot notation access
        for piece in pieces:
            chunks.append(Chunk(
                page_content=piece, # full text of the chunk
                metadata={"week": doc["week"], "day": doc["day"], "source": doc["source"]}
            ))

    print(f"Total chunks created: {len(chunks)}")
    print(f"\n--- Sample chunk ---\n")
    print(chunks[0].page_content)
    return chunks # A list of Chunk pydantic objects that include text, week, day, and source


def embed_and_store(chunks, embedder, collection):
    """
    Create page_content, metadata, and ids and stores it in variables
    Encodes the texts into vectors
    Adds the vector into the chroma database
    """
    texts = [chunk.page_content for chunk in chunks] # A list of all the text in chunks using dot notation
    metadatas = [chunk.metadata for chunk in chunks] # Creates metadata for the week, day, and the source using dot notation
    ids = [f"{chunk.metadata['source']}_{i}" for i, chunk in enumerate(chunks)] # Names each chunk with the file name and an index

    print("Embedding chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    # Adds the vectors into the chroma database with the following data
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
