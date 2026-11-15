from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv(override=True)

import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages



thread_id = '1'
config = {
    'configurable' : {'thread_id': thread_id}
}


class ChatState(TypedDict):

    messages : Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model='gpt-4o-mini')


def chat_node(state: ChatState):

    # take user query from state
    user_messages = state['messages']

    # send to llm
    response = llm.invoke(user_messages)

    # store the response back into state
    return {
        'messages': [response]
    }


checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

