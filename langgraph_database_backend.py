from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3

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


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)