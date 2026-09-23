from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# ── Define custom tools using @tool decorator ───────────────────────
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word."""
    return len(word)

# ── Model setup ─────────────────────────────────────────────────────
model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

tools = [add, multiply, get_word_length]
tool_map = {t.name: t for t in tools}

model_with_tools = model.bind_tools(tools)

# ── Ask a question that requires tool use ───────────────────────────
query = "What is 42 multiplied by 7? Also, how many letters are in the word 'intelligence'?"
messages = [HumanMessage(content=query)]

print(f"User: {query}\n")

response = model_with_tools.invoke(messages)
messages.append(response)

# ── Execute all requested tool calls ───────────────────────────────
if response.tool_calls:
    print(f"Model is calling {len(response.tool_calls)} tool(s):\n")
    for tool_call in response.tool_calls:
        print(f"  Tool : {tool_call['name']}")
        print(f"  Input: {tool_call['args']}")

        result = tool_map[tool_call["name"]].invoke(tool_call["args"])
        print(f"  Result: {result}\n")

        messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    # Final answer from the model
    final = model_with_tools.invoke(messages)
    content = final.content
    if isinstance(content, list):
        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    print("Assistant:", content)
else:
    print("Assistant:", response.content)
