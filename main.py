"""MCP server for the local LangGraph MCP client.

This server is launched by `langgraph_mcp_backend.py` via stdio:
  command: python3
  args: ["/Users/abodh_shah/Documents/repositories/langgraph/main.py"]

It exposes arithmetic (and related utility) tools over MCP.
"""

import os

import requests
from ddgs import DDGS
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(override=True)

mcp = FastMCP("arith")


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with the API key in the URL.
    """
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


if __name__ == "__main__":
    # Run as a stdio MCP server (expected by langgraph_mcp_backend.py).
    mcp.run(transport="stdio")
