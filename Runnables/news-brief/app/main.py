"""
News Brief — FastAPI backend

Wraps the original LangChain pipeline (Tavily search -> Gemini summarization)
behind a small JSON API, and serves the frontend UI.

Run with:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------------------------------------------------------------------
# Chain setup (same components as the original script)
# ---------------------------------------------------------------------------

search_tool = TavilySearchResults(max_results=5)

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Summarize the following search results into clear, concise bullet points.
Each bullet should be a single, self-contained fact or development. Do not
add a preamble or closing remarks — output only the bullet points.

Search results:
{news}

User question:
{question}
"""
)

chain = prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None


class SearchResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="News Brief API")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        raw_results = search_tool.invoke({"query": question})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc

    # Normalize Tavily results into a predictable shape for the frontend.
    sources: List[Source] = []
    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict):
                sources.append(
                    Source(
                        title=item.get("title"),
                        url=item.get("url"),
                        content=item.get("content"),
                    )
                )

    try:
        answer = chain.invoke({"news": raw_results, "question": question})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Summarization failed: {exc}"
        ) from exc

    return SearchResponse(question=question, answer=answer, sources=sources)


@app.get("/api/health")
def health():
    return {"status": "ok"}
