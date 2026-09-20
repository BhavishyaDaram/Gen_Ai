"""
streamlit_app.py

Web UI for the existing RAG pipeline (rag_core.py).

Run with:
    streamlit run streamlit_app.py

This does NOT change your architecture:
- same embedding model
- same Chroma DB
- same retriever settings
- same Gemini model and prompt
- API key still comes from .env
"""

import os
import time
import tempfile

import streamlit as st


# ---------------- Page config ----------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- Load RAG pipeline ----------------

try:
    from rag_core import get_answer, ingest_pdf

    RAG_READY = True
    RAG_ERROR = None

except Exception as e:
    RAG_READY = False
    RAG_ERROR = str(e)


# ---------------- Styling ----------------

st.markdown(
    """
    <style>
        .stChatMessage {
            border-radius: 12px;
        }

        .source-box {
            background-color: rgba(127,127,127,0.08);
            border-left: 3px solid #4B8BF5;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }

        .main-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 0.2rem;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Session state ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []


# ---------------- Sidebar ----------------

with st.sidebar:

    st.markdown("## 📄 RAG Assistant")

    st.caption(
        "Ask questions grounded in your document collection."
    )

    st.divider()

    st.markdown("**Pipeline**")

    st.markdown(
        "- Embeddings: `all-MiniLM-L6-v2`\n"
        "- Vector store: `Chroma` (local)\n"
        "- Retrieval: `MMR`, k=4, fetch_k=10\n"
        "- LLM: `gemini-3.6-flash`"
    )

    st.divider()

    st.markdown("**📤 Upload a document**")

    st.caption(
        "Add a PDF to the knowledge base — it becomes searchable immediately."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        disabled=not RAG_READY,
    )

    if uploaded_file is not None:

        already_done = (
            uploaded_file.name
            in st.session_state.ingested_files
        )

        if already_done:

            st.info(
                f"'{uploaded_file.name}' was already added this session.",
                icon="ℹ️",
            )

        else:

            if st.button(
                f"➕ Add '{uploaded_file.name}' to knowledge base",
                use_container_width=True,
            ):

                with st.spinner(
                    f"Reading and embedding {uploaded_file.name}..."
                ):

                    tmp_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf",
                        ) as tmp:

                            tmp.write(
                                uploaded_file.getbuffer()
                            )

                            tmp_path = tmp.name

                        num_chunks = ingest_pdf(
                            tmp_path,
                            source_name=uploaded_file.name,
                        )

                        st.session_state.ingested_files.append(
                            uploaded_file.name
                        )

                        st.success(
                            f"Added '{uploaded_file.name}' "
                            f"({num_chunks} chunks indexed).",
                            icon="✅",
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Failed to add '{uploaded_file.name}': {e}",
                            icon="🚫",
                        )

                    finally:

                        if tmp_path and os.path.exists(tmp_path):
                            os.remove(tmp_path)


    if st.session_state.ingested_files:

        with st.expander(
            f"📚 {len(st.session_state.ingested_files)} "
            "document(s) added this session"
        ):

            for fname in st.session_state.ingested_files:
                st.markdown(f"- {fname}")


    st.divider()

    show_sources = st.toggle(
        "Show retrieved sources",
        value=True,
    )


    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()

    if RAG_READY:

        st.success(
            "RAG pipeline connected",
            icon="✅",
        )

    else:

        st.error(
            "RAG pipeline not initialized",
            icon="⚠️",
        )


# ---------------- Main header ----------------

st.markdown(
    '<div class="main-header">'
    '<h1>💬 Document Q&A</h1>'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Answers are grounded only in the content of your indexed documents."
)


# ---------------- RAG initialization error ----------------

if not RAG_READY:

    st.error(
        "The RAG pipeline failed to initialize, so chat is disabled.\n\n"
        f"**Details:** {RAG_ERROR}\n\n"
        "Common fixes:\n"
        "- Make sure a `.env` file exists in the project root with "
        "`GEMINI_API_KEY=...`\n"
        "- Make sure the `chroma_db` folder exists in the project root "
        "and is populated\n"
        "- Make sure all packages from `requirements.txt` are installed",
        icon="🚫",
    )

    st.stop()


# ---------------- Chat history ----------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if (
            msg["role"] == "assistant"
            and show_sources
            and msg.get("sources")
        ):

            with st.expander(
                f"📚 {len(msg['sources'])} source chunk(s) used"
            ):

                for i, doc in enumerate(
                    msg["sources"],
                    start=1,
                ):

                    st.markdown(
                        f"""
                        <div class="source-box">
                            <b>Chunk {i}</b><br>
                            {doc}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ---------------- Empty state ----------------

if not st.session_state.messages:

    st.info(
        "Ask a question about your documents to get started.",
        icon="💡",
    )


# ---------------- Chat input ----------------

query = st.chat_input(
    "Ask a question about your documents..."
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
            "sources": None,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)


    with st.chat_message("assistant"):

        placeholder = st.empty()

        with st.spinner(
            "Retrieving context and generating answer..."
        ):

            try:

                answer, docs = get_answer(query)

            except Exception as e:

                answer = (
                    f"⚠️ Something went wrong while answering: {e}"
                )

                docs = []


        # ---------------- Normalize answer ----------------

        if isinstance(answer, list):

            text_parts = []

            for item in answer:

                if isinstance(item, dict):

                    text_parts.append(
                        item.get("text", "")
                    )

                else:

                    text_parts.append(
                        str(item)
                    )

            answer = "".join(text_parts)


        if not isinstance(answer, str):

            answer = str(answer)


        # ---------------- Display answer ----------------

        displayed = ""

        for word in answer.split():

            displayed += word + " "

            placeholder.markdown(displayed)

            time.sleep(0.01)


        # ---------------- Display sources ----------------

        if show_sources and docs:

            with st.expander(
                f"📚 {len(docs)} source chunk(s) used"
            ):

                for i, doc in enumerate(
                    docs,
                    start=1,
                ):

                    src_name = doc.metadata.get(
                        "source",
                        "unknown source",
                    )

                    st.markdown(
                        f"""
                        <div class="source-box">
                            <b>Chunk {i}</b> —
                            <i>{src_name}</i><br>
                            {doc.page_content}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


    # ---------------- Save assistant message ----------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [
                d.page_content for d in docs
            ] if docs else None,
        }
    )