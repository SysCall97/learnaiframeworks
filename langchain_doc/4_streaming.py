import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

model_name = "openai/gpt-oss-120b"
api_key = os.getenv("OPENAI_API_KEY")
openai_api_base="https://api.groq.com/openai/v1"

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.utils.uuid import uuid7


llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=openai_api_base
)

agent = create_agent(model=llm)

from langchain.messages import AIMessage, HumanMessage

for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
}, stream_mode="values"):
    # Each chunk contains the full state at that point
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")