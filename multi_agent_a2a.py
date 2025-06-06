"""Demonstrate two ADK agents communicating via A2A."""

import asyncio
import httpx

from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runner import Runner
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from a2a.client import (
    A2AClient,
    create_text_message_object,
    MessageSendParams,
    SendMessageRequest,
)


def perform_task(detail: str) -> str:
    """Tool executed by the remote worker agent."""
    return f"Task '{detail}' completed."


# Worker agent definition (mirrors the remote server)
worker_agent = LlmAgent(
    name="worker_agent",
    description="Handles delegated work.",
    model=LiteLlm(model="openai/gpt-4o"),
    instruction="Use perform_task to handle the request.",
    tools=[perform_task],
)


async def delegate_via_a2a(task: str) -> str:
    """Send a task to the worker agent via A2A."""
    async with httpx.AsyncClient() as http_client:
        client = A2AClient(httpx_client=http_client, url="http://localhost:8000/")
        message = create_text_message_object(content=task, role="user")
        params = MessageSendParams(message=message)
        request = SendMessageRequest(params=params)
        response = await client.send_message(request)
        part = response.root.result.parts[0]
        return part.text if hasattr(part, "text") else ""


# Coordinator agent delegates tasks via the A2A tool
delegate_tool = FunctionTool(func=delegate_via_a2a)
coordinator = LlmAgent(
    name="coordinator",
    description="Delegates work to a remote worker using A2A.",
    model=LiteLlm(model="openai/gpt-4o"),
    instruction="Use the delegate_via_a2a tool whenever a user requests work.",
    tools=[delegate_tool],
)


async def main() -> None:
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    runner = Runner(
        app_name="a2a_demo",
        agent=coordinator,
        artifact_service=artifact_service,
        session_service=session_service,
    )
    session = await session_service.create_session(app_name="a2a_demo", user_id="user")
    user_message = types.Content(role="user", parts=[types.Part.from_text("demo task")])
    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=user_message,
    ):
        if event.content.parts and hasattr(event.content.parts[0], "text"):
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())

