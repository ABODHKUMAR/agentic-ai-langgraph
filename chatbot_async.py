import asyncio
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(override=True)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o-mini")


# ****************************************** Tools **********************************************
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


tools = [calculator]
llm_with_tools = llm.bind_tools(tools)


def build_graph():
    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools, handle_tool_errors=True)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")

    # If the LLM asked for a tool, go to ToolNode; otherwise finish.
    graph.add_conditional_edges("chat_node", tools_condition)

    # After the tool runs, return to the LLM so it can answer.
    graph.add_edge("tools", "chat_node")

    return graph.compile()


async def main():
    chatbot = build_graph()

    result = await chatbot.ainvoke(
        {"messages": [HumanMessage(content="What is 25 multiplied by 4?")]}
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
