from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import json
import asyncio
import nest_asyncio
import os
import warnings
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain.chat_models import init_chat_model

# Suppress warnings
warnings.filterwarnings("ignore", message=".*additionalProperties.*")
warnings.filterwarnings("ignore", message=".*Key.*additionalProperties.*")
warnings.filterwarnings("ignore", category=UserWarning, module="mcp")
warnings.filterwarnings("ignore", category=Warning, module="mcp")
warnings.filterwarnings("ignore", message=".*ignoring.*")

nest_asyncio.apply()
load_dotenv()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY environment variable not set")
os.environ["GOOGLE_API_KEY"] = google_api_key

def clean_schema(schema: dict) -> dict:
    return {
        k: v for k, v in schema.items()
        if k not in ("$schema", "additionalProperties")
    }

class MCP_ChatBot:

    def __init__(self):
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)

            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])

            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": clean_schema(tool.inputSchema)
                })

        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)

        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def process_query(self, query: str):
        llm_with_tools = self.llm.bind_tools(self.available_tools)


        messages = [HumanMessage(content=query)]

        response = await llm_with_tools.ainvoke(messages)  # <-- CHANGED: use `await` and `ainvoke`
        print(response.content)

        if response.content and response.content.strip():
            messages.append(AIMessage(content=response.content))

        for tool_call in getattr(response, "tool_calls", []):  # safer access
            session = self.tool_to_session.get(tool_call["name"])
            if not session:
                print(f"No session found for tool: {tool_call['name']}")
                continue

            tool_result = await session.call_tool(
                tool_call["name"],
                arguments=tool_call["args"]
            )

            tool_msg = ToolMessage(
                tool_call_id=tool_call["id"],
                content=str(tool_result)
            )
            messages.append(tool_msg)

        final_response = await self.llm.ainvoke(messages)  # <-- again, async call
        print(final_response.content)

    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                await self.process_query(query)
                print("\n")
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
