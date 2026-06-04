import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

model = "openai/gpt-oss-120b"
api_key = os.getenv("OPENAI_API_KEY")
openai_api_base="https://api.groq.com/openai/v1"

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=openai_api_base
)

history = []

history.append(HumanMessage(content="My name is Rahim"))
response = llm.invoke(history)
history.append(AIMessage(response.content))

history.append(HumanMessage(content="What is my name?"))
response = llm.invoke(history)

history.append(AIMessage(response.content))

print(response.content)
print(history)