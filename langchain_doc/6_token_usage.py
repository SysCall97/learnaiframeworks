import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI


_ = load_dotenv(find_dotenv())

model_name = "openai/gpt-oss-120b"
api_key = os.getenv("OPENAI_API_KEY")
openai_api_base="https://api.groq.com/openai/v1"

llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=openai_api_base
)

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler

callback = UsageMetadataCallbackHandler()

agent = create_agent(model=llm)

result = agent.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"callbacks": [callback]}
)

print(callback.usage_metadata)