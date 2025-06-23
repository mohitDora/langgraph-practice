from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

import asyncio
import os

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "server": {
                "command": "node",
                "args": ["C:/projects/mcp-practice/dist/index.js"],
                "transport": "stdio",
                "env": {
                    "NOTION_API_KEY": "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",
                    "WEATHER_API_KEY": "e4605017f90d425bae751237252005",
                },
            }
        }
    )

    llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)
    res = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Add a new page in database with id 1f9d43ae6ddb809cab95e056fd8d8ba4 and title 'mohit'.",
                }
            ]
        }
    )
    print(res["messages"][-1].content)


asyncio.run(main())
