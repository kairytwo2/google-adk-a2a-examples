"""A2A server exposing a worker agent built with Google ADK."""

import asyncio
import uvicorn

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution.simple_request_context_builder import SimpleRequestContextBuilder
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.events.in_memory_queue_manager import InMemoryQueueManager
from a2a.server.apps.starlette_app import A2AStarletteApplication
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    TaskStatus,
    TaskState,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)
from a2a.utils import new_task, new_text_artifact


def perform_task(detail: str) -> str:
    """Tool executed by the worker agent."""
    return f"Task '{detail}' completed."


class ADKAgentExecutor(AgentExecutor):
    """Wrap an ADK LlmAgent for the A2A server."""

    def __init__(self, agent: LlmAgent) -> None:
        self.agent = agent

    async def execute(self, context: RequestContext, queue: EventQueue) -> None:
        user_input = context.get_user_input()
        events = await self.agent.run_async(user_input)
        text = ""
        for event in events:
            if event.content and event.content.parts:
                part = event.content.parts[0]
                if hasattr(part, "text"):
                    text = part.text
        task = context.current_task or new_task(context.message)
        queue.enqueue_event(task)
        queue.enqueue_event(
            TaskArtifactUpdateEvent(
                append=False,
                contextId=task.contextId,
                taskId=task.id,
                lastChunk=True,
                artifact=new_text_artifact(
                    name="result",
                    description="Task result",
                    text=text,
                ),
            )
        )
        queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(state=TaskState.completed),
                final=True,
                contextId=task.contextId,
                taskId=task.id,
            )
        )

    async def cancel(self, context: RequestContext, queue: EventQueue) -> None:
        queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(state=TaskState.canceled),
                final=True,
                contextId=context.context_id,
                taskId=context.task_id,
            )
        )


async def main() -> None:
    worker = LlmAgent(
        name="worker_agent",
        description="Handles delegated work.",
        model=LiteLlm(model="openai/gpt-4o"),
        instruction="Use perform_task to handle the request.",
        tools=[perform_task],
    )

    executor = ADKAgentExecutor(worker)
    task_store = InMemoryTaskStore()
    queue_manager = InMemoryQueueManager()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        queue_manager=queue_manager,
        request_context_builder=SimpleRequestContextBuilder(task_store=task_store),
    )
    card = AgentCard(
        name="worker_agent",
        description="ADK worker agent",
        capabilities=AgentCapabilities(),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        url="http://localhost:8000/",
        version="1.0",
    )
    app = A2AStarletteApplication(card, handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    asyncio.run(main())
