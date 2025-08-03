# MCP Research Project

A Model Context Protocol (MCP) project for AI research and paper analysis using agentic LLMs.

## Features

- **Research Server**: Automated arXiv paper search and analysis
- **MCP Chatbot**: Interactive AI assistant with multiple MCP server connections
- **Paper Management**: Organized storage and retrieval of research papers
- **Agentic LLM Integration**: Advanced AI capabilities for research tasks

## Project Structure

```
mcp_project/
├── research_server.py      # MCP server for arXiv research
├── mcp_chatbot.py         # Interactive AI chatbot
├── server_config.json      # MCP server configuration
├── requirements.txt        # Python dependencies
├── test_1.txt            # Research findings summary
├── papers/               # Downloaded research papers (gitignored)
└── .gitignore           # Git ignore rules
```

## Setup

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file with:

   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. **Run the research mcp server**:

   ```bash
   python research_server.py
   ```

4. **Run the chatbot client**:
   ```bash
   python mcp_chatbot.py
   ```

## MCP Servers

This project uses multiple MCP servers:

- **Research Server(internal)**: Handles arXiv paper searches and analysis
- **Filesystem Server(external)**: File system operations
- **Fetch Server(external)**: Web content fetching

## Try out yourself

Example prompt: fetch deeplearning.ai and find an interesting term to search papers around and then summarize your findings and write(use filesystem mcp) them to a file called test_1.txt
