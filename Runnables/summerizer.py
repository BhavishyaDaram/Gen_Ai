from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearch(max_results=2)
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Summarize the following news into clear bullet points.

{news}
"""
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.invoke({
    "query": "Latest AI news of 2026"
})

result = chain.invoke({
    "news": news_result
})

print(result)

print(search_tool.description)
print(search_tool.name)
print(search_tool.args)