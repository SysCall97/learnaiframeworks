import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

model_name = "openai/gpt-oss-120b"
api_key = os.getenv("OPENAI_API_KEY")
openai_api_base="https://api.groq.com/openai/v1"




from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI



llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=openai_api_base
)

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is LangChain?"
            }
        ]
    }
)

print(response["messages"][-1].content)