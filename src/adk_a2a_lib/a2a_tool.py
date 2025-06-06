"""A2ATool: ADK tool to delegate calls to a remote A2A agent."""

from __future__ import annotations

from typing import Any

import httpx
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from typing_extensions import override

from a2a.client import (
    A2AClient,
    create_text_message_object,
    MessageSendParams,
    SendMessageRequest,
)


class A2ATool(BaseTool):
    """Tool that sends a request to a remote A2A server."""

    _client: A2AClient

    def __init__(self, *, client: A2AClient, name: str = "a2a_tool", description: str = "Delegate to remote agent") -> None:
        super().__init__(name=name, description=description)
        self._client = client

    @classmethod
    async def from_url(cls, *, url: str, httpx_client: httpx.AsyncClient | None = None, name: str = "a2a_tool", description: str = "Delegate to remote agent") -> "A2ATool":
        """Convenience initializer using just a URL."""
        httpx_client = httpx_client or httpx.AsyncClient()
        client = A2AClient(httpx_client=httpx_client, url=url)
        return cls(client=client, name=name, description=description)

    @override
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        if not (content := args.get("task") or args.get("message") or args.get("input")):
            raise ValueError("A2ATool requires a 'task', 'input', or 'message' argument")

        message = create_text_message_object(content=content, role="user")
        params = MessageSendParams(message=message)
        request = SendMessageRequest(params=params)
        response = await self._client.send_message(request)

        part = response.root.result.parts[0]
        return part.text if hasattr(part, "text") else ""
