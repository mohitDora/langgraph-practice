from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.types import interrupt

load_dotenv()


@tool
def human_assistant(query: str) -> str:
    """Request assistant from human.

    Args:
        query (str): The query to search for.

    Returns:
        str: returns the response from human

    """
    human_feedback = interrupt({"query": query})
    return human_feedback["data"]


def search(query: str) -> dict:
    """
    Search for a query on Tavily.

    Args:
        query (str): The query to search for.

    Returns:
        dict: A dictionary containing the search results.
    """
    tavily = TavilySearch(max_results=2)
    return tavily.invoke(query)


tools = [human_assistant, search]
