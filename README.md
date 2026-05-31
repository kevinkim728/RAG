# RAG Study Assistant

**[Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/kevinkim728/RAG-study-assistant)**

A Retrieval-Augmented Generation (RAG) pipeline built as a personal study assistant for an LLM engineering course. The project includes a full evaluation framework used to systematically compare retrieval strategies and find the best-performing pipeline configuration.

## Architecture

The hybrid pipeline runs in four stages:

1. **Bi-encoder retrieval** — nomic-ai/nomic-embed-text-v1.5 encodes the query and retrieves the top 20 candidate chunks from ChromaDB
2. **Query rewriting** — an LLM rewrites the original query into a more precise search query, retrieves another 20 chunks, and the two result sets are merged
3. **Cross-encoder reranking** — BAAI/bge-reranker-large scores each candidate chunk against the query and cuts the pool down to 15
4. **LLM reranking** — gpt-5.4 re-orders the top 15 chunks by relevance and returns the final 10 for answer generation

## Setup

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Add API key** — create a `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
                or
   GROQ_API_KEY=your_key_here
   ```

3. **Ingest transcripts** — embed and store all course transcripts into ChromaDB:
   ```bash
   uv run ingest.py
   ```

4. **Launch the chat UI:**
   ```bash
   uv run app.py
   ```

## ** EXTRAS **

## Evaluation Results

Used Mean Reciprocal Rank (MRR) to measure how highly the most relevant result is ranked — max score is 1. The higher the better. Each row adds one stage to the pipeline, showing the cumulative impact on retrieval quality.


| Pipeline | What it does | MRR |
|---|---|---|
| Baseline | Vector search only | 0.453 |
| LLM Reranker | + query rewriting + LLM reranker | 0.705 |
| Cross Encoder | + cross encoder reranker | 0.694 |
| Hybrid | + cross encoder → LLM reranker | **0.759** |

### Key findings

- Improved baseline retrieval results by 67% through iterative pipeline improvements
- The hybrid approach performed best by combining a cross encoder to filter chunks with an LLM to make the final ranking decision
- The free Groq model (gpt-oss-120b) scored surprisingly close to gpt-5.4, with only a 0.032 MRR gap

## Stack

- **Embeddings**: [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **Cross encoder**: [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
- **Vector store**: ChromaDB
- **LLM**: gpt-5.4 (OpenAI)
- **UI**: Gradio
