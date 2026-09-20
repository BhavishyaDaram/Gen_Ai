import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

load_dotenv(ENV_PATH)

if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"


def check_api_key():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def get_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            "Chroma database not found. Run create_database.py first."
        )

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )


def get_retriever():
    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


def get_llm():
    check_api_key()

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


def ingest_pdf(file_path: str, source_name: str | None = None):
    loader = PyPDFLoader(file_path)

    pages = loader.load()

    chunks = text_splitter.split_documents(pages)

    display_name = source_name or os.path.basename(file_path)

    for chunk in chunks:
        chunk.metadata["source"] = display_name

    if not chunks:
        return 0

    vectorstore = get_vectorstore()

    vectorstore.add_documents(chunks)

    return len(chunks)


def get_answer(query: str):
    if not query or not query.strip():
        return "Please enter a question.", []

    retriever = get_retriever()

    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    llm = get_llm()

    response = llm.invoke(final_prompt)

    answer = response.content

    if isinstance(answer, list):
        text_parts = []

        for item in answer:
            if isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))

        answer = "".join(text_parts)

    if not isinstance(answer, str):
        answer = str(answer)

    return answer, docs