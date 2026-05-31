import gradio as gr
from answer import fetch_context_hybrid, generate_answer


def format_chunks(chunks):
    return "\n\n---\n\n".join(
        f"**Chunk {i+1}** *(Week: {c.metadata.get('week', 'N/A')}, {c.metadata.get('day', 'N/A')})*\n\n{c.page_content}"
        for i, c in enumerate(chunks)
    )


def chat(history):
    try:
        content = history[-1]["content"]
        query = content if isinstance(content, str) else content[0]["text"]
        prior = history[:-1]
        chunks = fetch_context_hybrid(query, history=prior)
        answer = generate_answer(query, chunks, history=prior)
        history.append({"role": "assistant", "content": answer})
        return history, format_chunks(chunks)
    except Exception as e:
        history.append({"role": "assistant", "content": f"Something went wrong: {e}"})
        return history, ""


def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]


theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

with gr.Blocks(title="RAG Study Assistant") as demo:
    gr.Markdown("# RAG Study Assistant\nAsk me anything about the LLM engineering course!")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=600)
            msg = gr.Textbox(placeholder="Ask a question about the LLM course...", show_label=False)
            gr.Examples(
                label="Example Questions",
                examples_per_page=6,
                examples=[
                    "What is the rule of thumb for converting tokens to words?",
                    "What is the key difference between RAG and fine-tuning?",
                    "Why is Q LoRA used for fine tuning instead of building a model from scratch?",
                    "What is an end point in the context of API calls?",
                    "What is the five step strategy for solving a business problem with AI?",
                    "Why are output tokens including reasoning more expensive to generate?",
                ],
                inputs=msg,
            )
        with gr.Column(scale=2):
            chunks_display = gr.Markdown(label="Retrieved Chunks", value="*Retrieved chunks will appear here*", container=True, height=600)
            gr.ClearButton(value="Clear Context", components=[chatbot, chunks_display])

    msg.submit(put_message_in_chatbot, inputs=[msg, chatbot], outputs=[msg, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, chunks_display]
    )

demo.launch(inbrowser=True, theme=theme)
