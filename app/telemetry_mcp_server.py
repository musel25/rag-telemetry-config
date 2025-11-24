#!/usr/bin/env python
import asyncio
from typing import Any, Dict

from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    CallToolRequest,
    CallToolResult,
)

from telemetry_rag import run_rag_telemetry_query

server = Server("telemetry-rag-mcp")


# ---------------------------------------------------------------------
# Tools definition
# ---------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_telemetry_config",
            description=(
                "Generate Cisco IOS XR telemetry configuration from a natural "
                "language query using RAG over YANG sensor paths."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": (
                            "Natural language query describing telemetry "
                            "requirements (protocol, server, interval, etc)."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve from Qdrant.",
                        "default": 8,
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Qdrant collection to query.",
                        "default": "fixed_window_embeddings",
                    },
                },
                "required": ["user_query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(request: CallToolRequest) -> CallToolResult:
    """
    Dispatch MCP tool calls to the underlying RAG functions.
    """
    if request.name == "generate_telemetry_config":
        args: Dict[str, Any] = request.arguments or {}

        user_query: str = args.get("user_query", "")
        top_k: int = int(args.get("top_k", 8))
        collection_name: str = args.get("collection_name", "fixed_window_embeddings")

        result = run_rag_telemetry_query(
            user_query=user_query,
            top_k=top_k,
            collection_name=collection_name,
        )

        # Return as TextContent (MCP transport handles JSON-string)
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json_dumps_pretty(result),
                )
            ]
        )

    # Unknown tool
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Unknown tool: {request.name}",
            )
        ]
    )


def json_dumps_pretty(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False)


async def main() -> None:
    """
    Run MCP server on stdio. This is the usual way MCP servers are run.
    """
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
