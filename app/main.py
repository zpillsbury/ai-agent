from typing import Annotated, TypedDict

import anyio
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .utilities.settings import settings

llm = ChatOpenAI(
    openai_api_key=settings.openai_key,
    model_name="gpt-4o-mini",
    max_retries=2,
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph_builder = StateGraph(State)


async def chatbot(state: State) -> State:
    """
    Chatbot
    """
    response_message = await llm.ainvoke(state["messages"])

    return {"messages": [response_message]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


async def main() -> None:
    """
    AI Agent
    """
    async for event in graph.astream({"messages": [("user", "how are you?")]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

    return None


if __name__ == "__main__":
    anyio.run(main)
