"""A2ARemoteAgent: wrapper to use a remote A2A server as a sub-agent."""

from __future__ import annotations

from typing import AsyncGenerator

import httpx
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing_extensions import override

from a2a.client import (
    A2AClient,
    create_text_message_object,
    MessageSendParams,
    SendMessageRequest,
)


class A2ARemoteAgent(BaseAgent):
    """An ADK agent that delegates execution to a remote A2A agent."""

    _client: A2AClient

    def __init__(self, *, client: A2AClient, name: str, description: str = "Remote agent") -> None:
        super().__init__(name=name, description=description)
        self._client = client

    @classmethod
    async def from_url(cls, *, url: str, httpx_client: httpx.AsyncClient | None = None, name: str, description: str = "Remote agent") -> "A2ARemoteAgent":
        httpx_client = httpx_client or httpx.AsyncClient()
        client = A2AClient(httpx_client=httpx_client, url=url)
        return cls(client=client, name=name, description=description)

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        user_parts = ctx.user_content.parts if ctx.user_content else []
        text = user_parts[0].text if user_parts and hasattr(user_parts[0], "text") else ""

        message = create_text_message_object(content=text, role="user")
        params = MessageSendParams(message=message)
        request = SendMessageRequest(params=params)
        response = await self._client.send_message(request)
        part = response.root.result.parts[0]
        content_text = part.text if hasattr(part, "text") else ""
        yield Event(invocation_id=ctx.invocation_id, author=self.name, branch=ctx.branch, content=ctx.user_content)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=create_text_message_object(role="agent", content=content_text),
        )
