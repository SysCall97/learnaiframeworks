import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

model_name = "openai/gpt-oss-120b"
api_key = os.getenv("OPENAI_API_KEY")
openai_api_base="https://api.groq.com/openai/v1"


from langchain_openai import ChatOpenAI
import subprocess

from langchain.agents import create_agent
from langchain.tools import tool

@tool
def customBash(*, command: str, restart: bool = False, **kw):
    """Execute a single concise bash command.

        Rules:
        - Never repeat arguments.
        - Never generate loops.
        - Prefer ls, cat, grep.
        - Keep command under 100 characters.
    """
    if restart:
        return "Bash session restarted"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {e}"

llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=openai_api_base
)

agent = create_agent(
    model=llm,
    tools=[customBash],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "if there is any python file in the directory starting with 2_ give me the content of that file"}]}
)

print(result)
print(result["messages"][-1].content)