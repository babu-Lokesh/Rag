
from langchain_openai import ChatOpenAI
from constants import MODEL,TEMPERATURE,OPENAI_API_KEY

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model = MODEL,temperature=TEMPERATURE,api_key = OPENAI_API_KEY )
