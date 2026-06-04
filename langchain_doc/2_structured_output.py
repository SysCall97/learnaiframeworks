import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

model_name = "openai/gpt-oss-120b"
api_key = os.getenv("OPENAI_API_KEY")
openai_api_base="https://api.groq.com/openai/v1"

from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

class Answer(BaseModel):
    summary: str
    confidence: float

llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=openai_api_base
)

agent = create_agent(model=llm, tools=[], response_format=Answer)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
print(result)
print(result["structured_response"].summary)


