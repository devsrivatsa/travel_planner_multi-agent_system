from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
import os

async def get_tools_async():
    """Get tools from the flight search mcp server"""
    tools, exit_stack = None, None
    try:
        server_params = StdioServerParameters(
            command="mcp-flight-search",
            args=["--connection_type", "stdio"],
            env={"SERP_API_KEY": os.getenv("SERP_API_KEY")}
        )
        tools, exit_stack = await MCPToolset.from_server(
            connection_params=server_params
        )
    except Exception as e:
        print(f"Error getting tools: {e}")
    
    return tools, exit_stack

async def get_agent_async():
    """Create an ADK agent equipped with tools from the MCP Server"""
    tools, exit_stack = await get_tools_async()
    
    root_agent = LlmAgent(
        model = "gemeni-2.5-pro-preview-03-25",
        name =  "flight_search_assistant",
        instruction = "Help user to search for flights using the available tools based on the prompt. If return date is not specified, use an empty string for one-way trips.",
        tools = tools
    )

    return root_agent, exit_stack


