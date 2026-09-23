import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "topic": "Machine Learning"
})

print(result)