from typing_extensions import TypedDict
from typing import Annotated
import os
from dotenv import load_dotenv
from langgraph.types import Command

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from tools import tools

load_dotenv()

memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

system_message = SystemMessage(
    content=(
        "You are a helpful AI assistant. "
        "You have access to tools only when necessary. "
        "Do not use tools for simple conversational greetings or questions that do not require external information. "
        "Only use the provided tools if the user's query explicitly requires searching for information or getting weather details."
    )
)

llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
llm_with_tools = llm.bind_tools(tools)


def chat(state: State):
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}


graph_builder.add_node("chat-node", chat)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chat-node")
graph_builder.add_conditional_edges("chat-node", tools_condition)
graph_builder.add_edge("tools", "chat-node")

graph = graph_builder.compile(checkpointer=memory)

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = "graph.png"
    filepath = os.path.join(script_dir, filename)

    if not os.path.exists(filepath):
        img_data = graph.get_graph().draw_mermaid_png()
        with open(filepath, "wb") as f:
            f.write(img_data)

        print(f"Saved to: {filepath}")
    else:
        pass

except Exception as e:
    import traceback

    traceback.print_exc()

config = {"configurable": {"thread_id": "2"}}

input = {"messages": "I need assistance."}

events = graph.stream(input, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

human_res = "We are the expert. We have the final result. Please provide the final result of the match."

human_command = Command(resume={"data": human_res})

events = graph.stream(human_command, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
