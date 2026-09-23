from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, ToolMessage

# ── Model + Tool setup ──────────────────────────────────────────────
model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

search_tool = TavilySearch(max_results=2)
tools = [search_tool]

# Bind tools to the model so it knows what it can call
model_with_tools = model.bind_tools(tools)

# ── Step 1: Ask the model a question ───────────────────────────────
query = "What are the latest AI developments in 2026?"
messages = [HumanMessage(content=query)]

print(f"User: {query}\n")

response = model_with_tools.invoke(messages)
messages.append(response)

# ── Step 2: Execute any tool calls the model requested ─────────────
if response.tool_calls:
    print(f"Model wants to call {len(response.tool_calls)} tool(s):\n")
    for tool_call in response.tool_calls:
        print(f"  Tool: {tool_call['name']}")
        print(f"  Args: {tool_call['args']}\n")

        # Run the tool
        tool_result = search_tool.invoke(tool_call["args"])

        # Feed result back as a ToolMessage
        messages.append(
            ToolMessage(
                content=str(tool_result) if tool_result else "No results found.",
                tool_call_id=tool_call["id"]
            )
        )

    # ── Step 3: Let model generate final answer with tool results ───
    final_response = model_with_tools.invoke(messages)
    content = final_response.content
    # Gemini may return a list of content blocks — extract text
    if isinstance(content, list):
        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    print("Assistant:", content)
else:
    # Model answered directly without needing a tool
    print("Assistant:", response.content)
