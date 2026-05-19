from answer import fetch_context_hybrid, generate_answer

if __name__ == "__main__":
    query = "Whats there to love about rag and whats not to love?"
    print("\n=== Result ===")
    chunks = fetch_context_hybrid(query)
    answer = generate_answer(query, chunks)
    print(answer)
