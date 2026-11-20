import os
import sqlite3
from typing import Annotated, TypedDict

import requests
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(override=True)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o-mini")


# ****************************************** Tools **********************************************
@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo.

    Tries multiple DuckDuckGo backends so a single failing backend
    does not break the workflow.
    """
    last_error = None
    for backend in ("google", "bing", "auto", "yahoo"):
        try:
            results = DDGS().text(
                query,
                region="us-en",
                max_results=5,
                backend=backend,
            )
            if results:
                return "\n\n".join(
                    f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
                    for r in results
                )
        except Exception as e:
            last_error = e
    return f"Search failed: {last_error}"


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers.

    Supported operations: add, sub, mul, div.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Cannot divide by zero"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation: {operation}"}

        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with the API key in the URL.
    """
    # Free Alpha Vantage keys allow ~25 requests/day.
    # Get your own key at https://www.alphavantage.co/support/#api-key
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "KJ7MNUC4W75DNQ62")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}


tools = [get_stock_price, search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)


# ****************************************** Graph **********************************************
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools, handle_tool_errors=True)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

# If the LLM asked for a tool, go to ToolNode; otherwise finish.
graph.add_conditional_edges("chat_node", tools_condition)

# After the tool runs, return to the LLM so it can answer.
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)
