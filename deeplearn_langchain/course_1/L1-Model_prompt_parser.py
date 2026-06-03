import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

_ = load_dotenv(find_dotenv())

# model = "llama-3.1-8b-instant"
model = "openai/gpt-oss-120b"
api_key = os.getenv("OPENAI_API_KEY")
openai_api_base="https://api.groq.com/openai/v1"

client = OpenAI(
    api_key=api_key,
    base_url=openai_api_base
)

def get_completion(prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

# print(get_completion("What is 1+1?"))

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt_openai = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}. Just give the cooresponding response. Do not add any extra line in your responnse
text: ```{customer_email}```
"""

#print(prompt)
#print(get_completion(prompt=prompt_openai))



## USING LANGCHAIN
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt_template_string = """Translate the text \
that is delimited by triple backticks 
into a style that is {style}. Just give the cooresponding response. Do not add any extra line in your responnse
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(prompt_template_string)

#print(prompt_template.messages[0].prompt)
#print(prompt_template.messages[0].prompt.input_variables)

customer_messages = prompt_template.format_messages(
    style=style,
    text=customer_email
)

#print(customer_messages)

chat = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=openai_api_base
)

#customer_response = chat.invoke(customer_messages)

#print(customer_response.content)




"""
getting json out of a review
"""

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in fifteen days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

from pydantic import BaseModel, Field

class ReviewAnalysis(BaseModel):
    is_gift: bool = Field(
        description="Whether the item was purchased as a gift"
    )

    delivery_days: int = Field(
        description="Number of days to arrive, -1 if unknown"
    )

    price_value: list[str] = Field(
        description="Sentences discussing price or value"
    )

structured_llm = chat.with_structured_output(ReviewAnalysis)

result = structured_llm.invoke(customer_review)

print(type(result))
print(result)