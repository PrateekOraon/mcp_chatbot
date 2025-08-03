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
        # self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        # self.available_tools: List[ToolDefinition] = []
        # self.tool_to_session: Dict[str, ClientSession] = {}
        # Tools list required for Anthropic API
        self.available_tools = []
        # Prompts list for quick display 
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions: Dict[str, ClientSession] = {}

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
            # self.sessions.append(session)
            
            try:
                # List available tools
                response = await session.list_tools()
                tools = response.tools
                print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
                for tool in tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": clean_schema(tool.inputSchema)
                    })
            
                # List available prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        })
                # List available resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
            
            except Exception as e:
                print(f"Error {e}")

        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)

            print("Connected to all servers:", self.sessions)

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
            session = self.sessions[tool_call["name"]]
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

    async def get_resource(self, resource_uri: str):
        session = self.sessions[resource_uri] if resource_uri in self.sessions else None
        
        # Fallback for papers URIs - try any papers resource session
        if not session and resource_uri.startswith("papers://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break
            
        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return
        
        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Errory: {e}")
    
    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return
        
        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    print(f"    - {arg_name}")
    
    async def execute_prompt(self, prompt_name, args):
        """Execute a prompt with the given arguments."""
        session = self.sessions[prompt_name]
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return
        
        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content
                
                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(item.text if hasattr(item, 'text') else str(item) 
                                  for item in prompt_content)
                
                print(f"\nExecuting prompt '{prompt_name}'...")
                await self.process_query(text)
        except Exception as e:
            print(f"Errorw: {e}")
    
    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("Use @folders to see available topics")
        print("Use @<topic> to search papers in that topic")
        print("Use /prompts to list available prompts")
        print("Use /prompt <name> <arg1=value1> to execute a prompt")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
        
                if query.lower() == 'quit':
                    break
                
                # Check for @resource syntax first
                if query.startswith('@'):
                    # Remove @ sign  
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
                    continue
                
                # Check for /command syntax
                if query.startswith('/'):
                    parts = query.split()
                    command = parts[0].lower()
                    
                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue
                        
                        prompt_name = parts[1]
                        args = {}
                        
                        # Parse arguments
                        for arg in parts[2:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key] = value
                        
                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"Unknown command: {command}")
                    continue
                
                await self.process_query(query)
                    
            except Exception as e:
                print(f"\nErrorm: {str(e)}")
    
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
