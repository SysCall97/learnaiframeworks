import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

model_name = "openai/gpt-oss-120b"
api_key = os.getenv("OPENAI_API_KEY")
openai_api_base="https://api.groq.com/openai/v1"

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7


llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=openai_api_base
)

agent = create_agent(model=llm, checkpointer=InMemorySaver())


config1 = {"configurable": {"thread_id": str(uuid7())}}
config2 = {"configurable": {"thread_id": str(uuid7())}}

result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config1,
)

print(result1["messages"][-1].content)
print("--------------------------------------------------------------------------------------------------------------------------------------")

result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the longest sea beach?"}]},
    config=config2,
)

print(result2["messages"][-1].content)
print("--------------------------------------------------------------------------------------------------------------------------------------")

result3 = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config1,
)

print(result3["messages"][-1].content)
print("--------------------------------------------------------------------------------------------------------------------------------------")

# https://docs.langchain.com/oss/python/langchain/agents