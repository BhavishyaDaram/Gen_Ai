# Wire Brief

A small local web app around your original search-and-summarize script:
Tavily for live search, Gemini for summarization, served through a FastAPI
backend with a standalone HTML/CSS/JS frontend (no build step, no framework).

## Structure

```
news-brief/
├── app/
│   ├── main.py           FastAPI backend (search + summarize endpoint)
│   └── static/
│       └── index.html    Frontend UI
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your API keys:
   ```bash
   cp .env.example .env
   # then edit .env and fill in TAVILY_API_KEY and GOOGLE_API_KEY
   ```

3. Run the server from the `news-brief/` directory:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Open your browser to:
   ```
   http://127.0.0.1:8000
   ```

## How it works

- The frontend posts your question to `POST /api/search`.
- The backend runs `TavilySearchResults` to pull live results, then feeds
  them into the same `ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser`
  chain from the original script to produce bullet-point summary.
- The response (summary + normalized source list) is rendered in the UI,
  with each source shown as a clickable card.

## Notes

- CORS isn't needed since the frontend is served by the same FastAPI app.
- If a request fails (e.g. missing/invalid API key, rate limit), the UI
  surfaces the backend's error message directly rather than failing silently.
