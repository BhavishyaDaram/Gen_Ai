from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()

model = ChatOpenAI(
    model="gpt-5.6-luna"
)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Summarize the given text clearly and concisely."),
        ("human", "Summarize the following text:\n\n{data}")
    ]
)

prompt = template.format_prompt(
    data=docs[0].page_content
)

result = model.invoke(prompt)

print(result.content)