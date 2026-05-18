from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from openai import OpenAI

load_dotenv(override=True)

class Chunk(BaseModel):
    page_content: str
    metadata: dict

# embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
# embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma = PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("transcripts")

client = OpenAI()
model = "gpt-5.4-mini"
# client = Groq()
# model = "openai/gpt-oss-120b"



def generate_answer(query, chunks, history=[]):
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"""You are a helpful study assistant for an LLM engineering course. Answer the user's question using only the context provided. If the answer isn't in the context, say so. Context: {context}"""},
      ] + history + [
          {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


def rewrite_query(query, history=[]):
    """
    Calls the LLM to rewrite the query in a more clear and concise way
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"""You are a search query optimizer for a knowledge base of LLM engineering course transcripts.
            Rewrite the user's question into a short, precise search query most likely to surface relevant content.

            This is the conversation history so far: {history}

            Respond ONLY with the rewritten query, nothing else."""},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


def merge_chunks(chunks1, chunks2):
    merged = chunks1[:] # Everything from chunks1 which is a list of Chunk object
    existing = [chunk.page_content for chunk in chunks1] # Put the page_content of each object into a list

    # Checks if any chunk from chunk2 exists within the existing chunks
    for chunk in chunks2:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def rerank(query, chunks):
    user_prompt = f"The user has asked the following question:\n\n{query}\n\nRank all chunks by relevance, most relevant first.\n\n"

    # Enumerates the chunk with an ID and the page_content to prepare the LLM to choose
    for i, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {i + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply with ONLY the chunk IDs as comma-separated integers, most relevant first. Example: 3,1,4,2,5..."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a document re-ranker. Given a question and a list of chunks, return them ranked by relevance to the question, most relevant first. Respond in JSON format."},
            {"role": "user", "content": user_prompt}
        ],
    )
    order_str = response.choices[0].message.content.strip()
    order = [int(x.strip()) for x in order_str.split(',') if x.strip().isdigit()]
    order = [i for i in order if 1 <= i <= len(chunks)] # Filter out-of-range IDs the LLM may hallucinate
    print(f"Order returned by LLM: {order}")
    return [chunks[i - 1] for i in order] # Reorders chunks by using the ranked IDs from the LLM, converting from 1-indexed to 0-indexed


def fetch_context(query, n_results=20, history=[], final_k=10):
    query_embedding = embedder.encode(query).tolist() # Convert the query into a vector and puts it in a list
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results) # Uses the list of vectors from query_embedding and gives n_results of similar vectors
    chunks1 = [Chunk(page_content=doc, metadata=meta) for doc, meta in zip(results["documents"][0], results["metadatas"][0])] # Converts the chunk's page_content and metadata which was in a list, back into a Chunk object

    rewritten = rewrite_query(query, history) # Rewrites the query into a concise way
    rewritten_embedding = embedder.encode(rewritten).tolist() # Converts the rewritten query into a vector and puts it in a list
    results2 = collection.query(query_embeddings=[rewritten_embedding], n_results=n_results) # Uses the list of vectors from rewritten_embedding and gives n_results of similar vectors
    chunks2 = [Chunk(page_content=doc, metadata=meta) for doc, meta in zip(results2["documents"][0], results2["metadatas"][0])] # Converts the chunk's page_content and metadata which was in a list, back into a Chunk object

    merged = merge_chunks(chunks1, chunks2) # Eliminates duplicates from the chunks
    reranked = rerank(query, merged) # Ranks them in order from best to worst
    return reranked[:final_k] # Gives the final_k best chunks


def fetch_context_baseline(query, n_results=10):
    """
    Baseline fetch_context to get a baseline score for evaluation
    """
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return [Chunk(page_content=doc, metadata=meta) for doc, meta in zip(results["documents"][0], results["metadatas"][0])]


def fetch_context_crossencoder(query, n_results=20, final_k=10):
    """
    A fetch_context for a cross encoder technique
    """
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    chunks1 = [Chunk(page_content=doc, metadata=meta)
               for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

    rewritten = rewrite_query(query)
    rewritten_embedding = embedder.encode(rewritten).tolist()
    results2 = collection.query(query_embeddings=[rewritten_embedding], n_results=n_results)
    chunks2 = [Chunk(page_content=doc, metadata=meta) for doc, meta in zip(results2["documents"][0], results2["metadatas"][0])]

    merged = merge_chunks(chunks1, chunks2) # All the merged Chunk objects
    pairs = [[query, chunk.page_content] for chunk in merged] # pairs the query and each chunks page_content in their own list
    scores = cross_encoder.predict(pairs) # Uses cross encoder to predict the pairs. Returns a list of Tensor objects which will be a rank
    ranked = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True) # Zips the score with the chunk as a tuple and sorted from highest to lowest
    return [chunk for _, chunk in ranked[:final_k]]  # Returns final_k number of Chunk objects in a list


def fetch_context_hybrid(query, n_results=20, ce_k=15, final_k=10):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    chunks1 = [Chunk(page_content=doc, metadata=meta)
               for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

    rewritten = rewrite_query(query)
    rewritten_embedding = embedder.encode(rewritten).tolist()
    results2 = collection.query(query_embeddings=[rewritten_embedding], n_results=n_results)
    chunks2 = [Chunk(page_content=doc, metadata=meta)
               for doc, meta in zip(results2["documents"][0], results2["metadatas"][0])]

    merged = merge_chunks(chunks1, chunks2) # All merged Chunk objects
    pairs = [[query, chunk.page_content] for chunk in merged] # pairs the query and each chunks page_content in their own list
    scores = cross_encoder.predict(pairs) # Uses cross encoder to predict the pairs. Returns a list of Tensor objects which will be a rank
    ranked = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True) # Zips the score with the chunk as a tuple and sorted from highest to lowest
    ce_top = [chunk for _, chunk in ranked[:ce_k]]  # cross-encoder cuts pool to 15
    return rerank(query, ce_top)[:final_k]           # Returns final_k number of Chunk objects in a list
