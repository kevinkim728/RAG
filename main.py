from answer import fetch_context, fetch_context_baseline, fetch_context_crossencoder, fetch_context_hybrid, generate_answer

if __name__ == "__main__":
    query = "Whats there to love about rag and whats not to love?"

    print("=== Baseline ===")
    chunks = fetch_context_baseline(query)
    answer = generate_answer(query, chunks)
    print(answer)

    print("\n=== LLM Reranker ===")
    chunks = fetch_context(query)
    answer = generate_answer(query, chunks)
    print(answer)

    print("\n=== Cross Encoder ===")
    chunks = fetch_context_crossencoder(query)
    answer = generate_answer(query, chunks)
    print(answer)

    print("\n=== Hybrid ===")
    chunks = fetch_context_hybrid(query)
    answer = generate_answer(query, chunks)
    print(answer)
