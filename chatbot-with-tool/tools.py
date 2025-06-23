from langchain_tavily import TavilySearch
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_weather(city_name: str) -> dict:
    """
    Get the current weather in a city.

    Args:
        city_name (str): The name of the city.

    Returns:
        tuple: A dictionary containing the name of the city and the temperature in Celsius.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        raise EnvironmentError("WEATHER_API_KEY not set in environment variables.")

    url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": api_key,
        "q": city_name
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    name = data["location"]["name"]
    temp_c = data["current"]["temp_c"]

    return {"location":name, "temp_in_c":temp_c}

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

tools = [
    get_weather,
    search
]