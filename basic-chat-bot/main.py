from typing_extensions import TypedDict
from typing import Annotated
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

load_dotenv()

class State(TypedDict):
    messages:Annotated[list, add_messages]

graph_builder=StateGraph(State)

llm = ChatGroq(model="llama3-8b-8192",api_key=os.getenv("GROQ_API_KEY"))

def chat(state:State):
    return { "messages": [llm.invoke(state["messages"])] }

graph_builder.add_edge(START,"chat-node")
graph_builder.add_node("chat-node",chat)    
graph_builder.add_edge("chat-node",END)

graph=graph_builder.compile()

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

res=graph.invoke({"messages":"hello"})

print(res["messages"][-1].content)