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


embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
cross_encoder = CrossEncoder("BAAI/bge-reranker-large")
chroma = PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("transcripts")

client = OpenAI()
model = "gpt-5.4"
# client = Groq()
# model = "openai/gpt-oss-120b"



def rewrite_query(query, history=[]):
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
    merged = chunks1[:]
    existing = [chunk.page_content for chunk in chunks1]

    for chunk in chunks2:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def rerank(query, chunks):
    user_prompt = f"The user has asked the following question:\n\n{query}\n\nRank all chunks by relevance, most relevant first.\n\n"

    for i, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {i + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply with ONLY the chunk IDs as comma-separated integers, most relevant first. Example: 3,1,4,2,5..."

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a document re-ranker. Given a question and a list of chunks, return them ranked by relevance to the question, most relevant first. Return results as comma-separated integers only"},
            {"role": "user", "content": user_prompt}
        ],
    )
    order_str = response.choices[0].message.content.strip()
    order = [int(x.strip()) for x in order_str.split(',') if x.strip().isdigit()]
    order = [i for i in order if 1 <= i <= len(chunks)] # Filter out-of-range IDs the LLM may hallucinate
    print(f"Order returned by LLM: {order}")
    return [chunks[i - 1] for i in order] # LLM returns 1-indexed IDs


def fetch_context_hybrid(query, n_results=20, ce_k=15, final_k=10, history=[]):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    chunks1 = [Chunk(page_content=doc, metadata=meta)
               for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

    rewritten = rewrite_query(query, history)
    rewritten_embedding = embedder.encode(rewritten).tolist()
    results2 = collection.query(query_embeddings=[rewritten_embedding], n_results=n_results)
    chunks2 = [Chunk(page_content=doc, metadata=meta)
               for doc, meta in zip(results2["documents"][0], results2["metadatas"][0])]

    merged = merge_chunks(chunks1, chunks2)
    pairs = [[query, chunk.page_content] for chunk in merged]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
    ce_top = [chunk for _, chunk in ranked[:ce_k]]
    return rerank(query, ce_top)[:final_k]


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