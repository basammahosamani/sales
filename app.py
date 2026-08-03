# 5. Gradio User Interface
with gr.Blocks(title="PragyanAI Intelligent Assistant") as demo:
    gr.Markdown("# PragyanAI Conversational Sales & FAQ Assistant")
    gr.Markdown("Answers program questions based on the **PragyanAI Presentation & FAQ Sheet**.")

    with gr.Row():
        with gr.Column(scale=1):
            persona_selector = gr.Dropdown(
                choices=list(SALES_PROMPTS.keys()),
                value="PragyanAI Student Counselor",
                label="Select PragyanAI Persona",
                interactive=True
            )
            file_uploader = gr.File(
                label="Upload Additional PDFs or Excel Sheets",
                file_count="multiple",
                file_types=[".pdf", ".xlsx", ".xls"]
            )
            upload_status = gr.Textbox(label="Knowledge Base Status", value="PragyanAI presentation FAQ pre-loaded.", interactive=False)
            file_uploader.change(fn=load_documents_into_vectorstore, inputs=[file_uploader], outputs=[upload_status])

        with gr.Column(scale=3):
            chatbot_ui = gr.ChatInterface(
                fn=respond,
                additional_inputs=[persona_selector]
            )
            clear_btn = gr.Button("Clear Memory for Selected Persona", variant="secondary")
            clear_btn.click(fn=clear_chat_history, inputs=[persona_selector], outputs=None)

if __name__ == "__main__":
    demo.launch
