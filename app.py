# ============================================================
# PragyanAI Conversational AI Assistant
# Streamlit + LangChain + FAISS + Groq
# Part 1/4
# ============================================================


import os
import tempfile
import streamlit as st
import pandas as pd


from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from langchain_community.document_loaders import (
    PyPDFLoader
)


from langchain_community.vectorstores import FAISS


from langchain_huggingface import (
    HuggingFaceEmbeddings
)


from langchain_groq import (
    ChatGroq
)


from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)



# ============================================================
# Streamlit Page Configuration
# ============================================================


st.set_page_config(
    page_title="PragyanAI Conversational AI Assistant",
    page_icon="🤖",
    layout="wide"
)



# ============================================================
# Title
# ============================================================


st.title(
    "🤖 PragyanAI Conversational AI Assistant"
)


st.write(
    """
    This AI assistant answers questions about the 
    PragyanAI AI/GenAI Program using the official knowledge base.
    """
)



# ============================================================
# Groq API Key
# ============================================================


try:

    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


except Exception:

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )



if not GROQ_API_KEY:

    st.error(
        """
        ❌ Groq API Key missing.

        Add GROQ_API_KEY in Streamlit Secrets.
        """
    )

    st.stop()



# ============================================================
# Personas
# ============================================================


PERSONAS = {


"PragyanAI Student Counselor":

"""
You are Aarav, PragyanAI Student Counselor.

Help students understand the AI/GenAI program.

Focus on:
- Program structure
- Curriculum
- Fees
- Career opportunities
- Projects

Answer only using provided knowledge context.

Be friendly and encouraging.
""",



"PragyanAI Institutional Advisor":

"""
You are Dr. Kavita, Institutional Relations Advisor.

Help colleges understand PragyanAI partnership programs.

Focus on:
- Industry readiness
- CoE model
- Student transformation
- Career pathways

Use only provided knowledge context.
""",



"PragyanAI Enterprise Placement Lead":

"""
You are Rohan, Enterprise Placement Lead.

Explain hiring opportunities and technical capabilities.

Focus on:
- AI Engineers
- GenAI Engineers
- Agentic AI Engineers
- Enterprise projects

Use only provided knowledge context.
"""

}




# ============================================================
# Default PragyanAI FAQ Knowledge
# ============================================================


FAQ_DATA = {


"Category":[

"Program Overview",
"Program Structure",
"Program Structure",
"Pricing & Fees",
"Pricing & Fees",
"Curriculum",
"Curriculum",
"Evaluation",
"Career",
"Leadership"

],



"Question":[


"What is PragyanAI program duration?",

"What happens in Phase 1?",

"What happens in Phase 2?",

"What is the fee structure?",

"What salary can students expect?",

"What modules are covered in Months 1-3?",

"What modules are covered in Months 4-6?",

"How are students evaluated?",

"What career roles are available?",

"Who leads PragyanAI?"

],



"Answer":[


"PragyanAI is an 18-month AI/GenAI program with 6 months offline training followed by 12 months internship and placement drive.",


"Phase 1 includes 6 months offline training with classroom sessions, hands-on labs, projects, hackathons and technical seminars.",


"Phase 2 includes internship, live client projects, mock interviews, resume building and startup/product development exposure.",


"Founding Batch fee is ₹50,000 training fee plus ₹50,000 success fee after placement.",


"Target packages: AI Engineer ₹6-15 LPA, GenAI Engineer ₹8-18 LPA, Agentic AI Engineer ₹10-25 LPA.",


"Months 1-3 cover Python Full Stack, Analytics, Data Science, BI Analytics, Machine Learning, AutoML and Streamlit deployment.",


"Months 4-6 cover Deep Learning, Computer Vision, NLP, Generative AI, LLMs, RAG, LangChain, CrewAI, AutoGen and Multi-Agent Systems.",


"Students are evaluated through technical seminars and 48-hour skill hackathons.",


"Career tracks include Data Analyst, Data Scientist, ML Engineer, AI Engineer, GenAI Engineer, Agentic AI Engineer and Product Engineer.",


"PragyanAI is led by Sateesh Ambesange, Co-Founder with 25+ years IT experience."

]

}



faq_df = pd.DataFrame(
    FAQ_DATA
)



# ============================================================
# Embedding Model
# ============================================================


@st.cache_resource
def load_embeddings():


    return HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"

    )



embeddings = load_embeddings()



# ============================================================
# End Part 1
# ============================================================
# ============================================================
# Part 2/4
# Document Loading + FAISS Vector Database
# ============================================================



# ============================================================
# Convert FAQ dataframe into Documents
# ============================================================


def load_default_faq_documents():

    documents = []


    for _, row in faq_df.iterrows():

        content = f"""

Category:
{row['Category']}

Question:
{row['Question']}

Answer:
{row['Answer']}

"""


        documents.append(

            Document(

                page_content=content,

                metadata={
                    "source":"PragyanAI FAQ"
                }

            )

        )


    return documents




# ============================================================
# Load Uploaded Documents
# ============================================================


def load_uploaded_documents(uploaded_files):


    documents = []



    if not uploaded_files:

        return documents




    for uploaded_file in uploaded_files:



        filename = uploaded_file.name.lower()



        # ----------------------------------------------------
        # PDF Upload
        # ----------------------------------------------------


        if filename.endswith(".pdf"):



            with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=".pdf"

            ) as temp_file:



                temp_file.write(

                    uploaded_file.getvalue()

                )


                temp_path = temp_file.name



            loader = PyPDFLoader(

                temp_path

            )


            pdf_docs = loader.load()



            documents.extend(

                pdf_docs

            )


            os.remove(

                temp_path

            )




        # ----------------------------------------------------
        # Excel Upload
        # ----------------------------------------------------


        elif filename.endswith(

            (".xlsx",".xls")

        ):



            excel_data = pd.read_excel(

                uploaded_file

            )



            for _, row in excel_data.iterrows():



                text = " | ".join(

                    [

                        f"{column}: {value}"

                        for column,value

                        in row.items()

                    ]

                )



                documents.append(

                    Document(

                        page_content=text,

                        metadata={

                            "source":filename

                        }

                    )

                )



    return documents





# ============================================================
# Create FAISS Vector Store
# ============================================================


@st.cache_resource(show_spinner=False)
def build_vectorstore(uploaded_files=None):


    documents = []



    # Load default FAQ

    documents.extend(

        load_default_faq_documents()

    )



    # Load uploaded files

    if uploaded_files:


        documents.extend(

            load_uploaded_documents(

                uploaded_files

            )

        )




    # Split large documents


    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=150

    )



    final_documents = splitter.split_documents(

        documents

    )




    vector_db = FAISS.from_documents(

        final_documents,

        embeddings

    )



    return vector_db





# ============================================================
# Initialize Knowledge Base
# ============================================================


if "vectorstore" not in st.session_state:


    st.session_state.vectorstore = build_vectorstore()



# ============================================================
# End Part 2
# ============================================================
# ============================================================
# Part 3/4
# Groq LLM + RAG Pipeline + Memory
# ============================================================



# ============================================================
# Initialize Groq Model
# ============================================================


@st.cache_resource
def load_llm():


    return ChatGroq(

        api_key=GROQ_API_KEY,

        model="llama-3.1-8b-instant",

        temperature=0.3

    )



llm = load_llm()




# ============================================================
# Chat Memory
# ============================================================


if "messages" not in st.session_state:


    st.session_state.messages = []




# ============================================================
# Create Prompt
# ============================================================


def create_prompt(
    persona,
    context,
    question
):


    system_prompt = PERSONAS.get(

        persona,

        PERSONAS[
            "PragyanAI Student Counselor"
        ]

    )



    prompt = f"""

{system_prompt}


IMPORTANT RULES:

1. Answer only using the Context below.
2. Do not create fake information.
3. If information is not available, say:
   "Please contact PragyanAI team for more details."


CONTEXT:

{context}


USER QUESTION:

{question}


ANSWER:

"""



    return prompt





# ============================================================
# Generate Answer
# ============================================================


def generate_answer(

    question,

    persona

):



    vectorstore = (

        st.session_state.vectorstore

    )



    # Retrieve documents


    retriever = vectorstore.as_retriever(

        search_kwargs={

            "k":4

        }

    )



    relevant_docs = retriever.invoke(

        question

    )



    context = "\n\n".join(

        [

            doc.page_content

            for doc in relevant_docs

        ]

    )



    final_prompt = create_prompt(

        persona,

        context,

        question

    )



    response = llm.invoke(

        final_prompt

    )



    return response.content





# ============================================================
# Clear Chat Function
# ============================================================


def clear_chat():


    st.session_state.messages = []



# ============================================================
# End Part 3
# ============================================================
# ============================================================
# Part 4/4
# Streamlit User Interface
# ============================================================



# ============================================================
# Sidebar
# ============================================================


with st.sidebar:


    st.header(
        "⚙️ Settings"
    )


    selected_persona = st.selectbox(

        "Choose Assistant Persona",

        options=list(
            PERSONAS.keys()
        )

    )



    st.divider()



    uploaded_files = st.file_uploader(

        "Upload PDF / Excel Knowledge Files",

        type=[

            "pdf",

            "xlsx",

            "xls"

        ],

        accept_multiple_files=True

    )




    if st.button(

        "🔄 Update Knowledge Base"

    ):


        with st.spinner(

            "Building Knowledge Base..."

        ):


            st.session_state.vectorstore = (

                build_vectorstore(

                    uploaded_files

                )

            )


        st.success(

            "Knowledge Base Updated Successfully!"

        )




    st.divider()



    if st.button(

        "🗑 Clear Chat"

    ):


        clear_chat()


        st.rerun()





# ============================================================
# Display Chat History
# ============================================================


for message in st.session_state.messages:


    with st.chat_message(

        message["role"]

    ):


        st.write(

            message["content"]

        )





# ============================================================
# Chat Input
# ============================================================


user_question = st.chat_input(

    "Ask about PragyanAI program..."

)




if user_question:



    # Save user message


    st.session_state.messages.append(

        {

            "role":"user",

            "content":user_question

        }

    )



    with st.chat_message(

        "user"

    ):


        st.write(

            user_question

        )




    # Generate response


    with st.chat_message(

        "assistant"

    ):



        with st.spinner(

            "Thinking..."

        ):



            answer = generate_answer(

                user_question,

                selected_persona

            )



            st.write(

                answer

            )



    # Save assistant message


    st.session_state.messages.append(

        {

            "role":"assistant",

            "content":answer

        }

    )





# ============================================================
# Footer
# ============================================================


st.divider()


st.caption(

    "Powered by PragyanAI | LangChain + FAISS + Groq"

)
