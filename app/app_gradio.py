#!/usr/bin/env python
import gradio as gr

from telemetry_rag import run_rag_telemetry_query


def generate_config_ui(user_query: str, top_k: int, collection_name: str):
    if not user_query.strip():
        return "Please provide a query.", ""

    result = run_rag_telemetry_query(
        user_query=user_query,
        top_k=top_k,
        collection_name=collection_name,
    )

    config_text = result["config"]
    retrieved_chunks = result["retrieved_chunks"]

    # Build a small debug text for retrieved chunks
    chunks_str_lines = []
    for c in retrieved_chunks:
        chunks_str_lines.append(
            f"[score={c['score']:.4f}] {c['file_path']} (chunk {c['chunk_index']})"
        )
    chunks_str = "\n".join(chunks_str_lines)

    return config_text, chunks_str


with gr.Blocks() as demo:
    gr.Markdown("# Telemetry RAG for Cisco IOS XR (BGP, OSPF, etc.)")
    gr.Markdown(
        "Enter a natural language request for telemetry configuration. "
        "The app will retrieve YANG sensor paths from Qdrant and generate "
        "a valid IOS XR telemetry model-driven config."
    )

    with gr.Row():
        user_query = gr.Textbox(
            label="Telemetry request",
            lines=4,
            placeholder=(
                "Example: Generate telemetry configuration for Cisco IOS XR about BGP. "
                "Use gRPC with no TLS, telemetry server 192.0.2.0 port 57500. "
                "Choose relevant BGP sensor paths."
            ),
        )

    with gr.Row():
        top_k = gr.Slider(
            minimum=1,
            maximum=20,
            value=10,
            step=1,
            label="Number of chunks (top_k)",
        )
        collection_name = gr.Textbox(
            label="Qdrant collection name",
            value="fixed_window_embeddings",
        )

    generate_btn = gr.Button("Generate Telemetry Config")

    config_output = gr.Code(
        label="Generated IOS XR telemetry configuration",
        # language="bash",
    )
    chunks_output = gr.Textbox(
        label="Retrieved chunks (debug)",
        lines=10,
    )

    generate_btn.click(
        fn=generate_config_ui,
        inputs=[user_query, top_k, collection_name],
        outputs=[config_output, chunks_output],
    )


if __name__ == "__main__":
    demo.launch()
