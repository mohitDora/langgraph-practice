from typing_extensions import TypedDict
from typing import Annotated
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

from tools import tools

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

system_message = SystemMessage(
    content=(
        """
        {instructions}

        TOOLS:
        ------

        You have access to the following tools:

        {tools}

        To use a tool, please use the following format:

        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```

        Begin!

        Previous conversation history:
        {chat_history}

        New input: {input}
        {agent_scratchpad}

        """
    )
)

llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=os.getenv("GROQ_API_KEY"))
llm_with_tools = llm.bind_tools(tools)


def chat(state: State):
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}


graph_builder.add_node("chat-node", chat)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chat-node")
graph_builder.add_conditional_edges("chat-node", tools_condition)
graph_builder.add_edge("tools", "chat-node")

graph = graph_builder.compile()

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

# res = graph.invoke({"messages": "What is the final result of real madrid vs pachuca?"})

# for message in res["messages"]:
#     message.pretty_print()
