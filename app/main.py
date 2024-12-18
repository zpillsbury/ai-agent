from langchain_openai import ChatOpenAI

from .utilities.settings import settings

llm = ChatOpenAI(
    openai_api_key=settings.openai_key,
    model_name="gpt-4o-mini",
    max_retries=2,
)
r = llm.invoke("How are you?")
print(r.content)
