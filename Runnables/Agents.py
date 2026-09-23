from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


search_tool = TavilySearchResults(max_results=5)


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest"
)


prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Summarize the following search results into clear bullet points.

Search results:
{news}

User question:
{question}
"""
)


chain = prompt | llm | StrOutputParser()


question = input("Enter the news topic or question you want to search for: ")


news_result = search_tool.invoke({
    "query": question
})


result = chain.invoke({
    "news": news_result,
    "question": question
})


print("\n--- ANSWER ---\n")
print(result)


print("\n--- TOOL INFORMATION ---")
print(search_tool.description)
print(search_tool.name)
print(search_tool.args)