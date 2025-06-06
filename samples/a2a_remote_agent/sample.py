import asyncio
from adk_a2a_lib import A2ARemoteAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.runner import Runner
from google.genai import types

async def main() -> None:
    agent = await A2ARemoteAgent.from_url(name="remote", url="http://localhost:8000/")
    runner = Runner(
        app_name="remote_wrapper",
        agent=agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(app_name="remote_wrapper", user_id="user")
    user_message = types.Content(role="user", parts=[types.Part.from_text("demo")])
    async for event in runner.run_async(user_id="user", session_id=session.id, new_message=user_message):
        if event.content and event.content.parts and hasattr(event.content.parts[0], "text"):
            print(event.content.parts[0].text)

if __name__ == "__main__":
    asyncio.run(main())
