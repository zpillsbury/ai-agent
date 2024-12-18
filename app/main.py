from datetime import datetime, timezone
from typing import Annotated, TypedDict

import anyio
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .utilities.settings import settings

now = datetime.now(timezone.utc)

system_prompt = f"""
You are an AI assistant helping an user.

Current Date: {now}
"""

web_search = TavilySearchResults(max_results=2)
tools = [web_search]

llm = ChatOpenAI(
    openai_api_key=settings.openai_key,
    model_name="gpt-4o-mini",
    max_retries=2,
)

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph_builder = StateGraph(State)


async def chatbot(state: State) -> State:
    """
    Chatbot
    """
    response_message = await llm_with_tools.ainvoke(state["messages"])

    return {"messages": [response_message]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()


async def run_graph(question: str) -> None:
    """
    Run the graph
    """
    async for event in graph.astream(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ]
        }
    ):
        for value in event.values():
            print(value["messages"][-1].content)

    return None


async def main() -> None:
    """
    AI Agent
    """
    while True:
        question = input("q: ")
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        await run_graph(question)

    return None


if __name__ == "__main__":
    anyio.run(main)
