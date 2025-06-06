import sys
import types
import asyncio
import pytest
# stub httpx module
httpx = types.ModuleType("httpx")
class AsyncClient: pass
httpx.AsyncClient = AsyncClient
sys.modules.setdefault("httpx", httpx)


# ---- Stub dependencies for google.adk ----
class _BaseTool:
    def __init__(self, *, name, description, is_long_running=False):
        self.name = name
        self.description = description
        self.is_long_running = is_long_running

class _ToolContext:
    pass

class _BaseAgent:
    def __init__(self, *, name, description=""):
        self.name = name
        self.description = description

class _InvocationContext:
    def __init__(self, *, invocation_id, branch=None, user_content=None):
        self.invocation_id = invocation_id
        self.branch = branch
        self.user_content = user_content

class _Event:
    def __init__(self, *, invocation_id, author, branch=None, content=None):
        self.invocation_id = invocation_id
        self.author = author
        self.branch = branch
        self.content = content

# build google.adk package structure with minimal classes
_google = types.ModuleType("google")
_adk = types.ModuleType("google.adk")
_tools = types.ModuleType("google.adk.tools")
_tools_base = types.ModuleType("google.adk.tools.base_tool")
_tools_ctx = types.ModuleType("google.adk.tools.tool_context")
_agents = types.ModuleType("google.adk.agents")
_agents_base = types.ModuleType("google.adk.agents.base_agent")
_agents_ctx = types.ModuleType("google.adk.agents.invocation_context")
_events_mod = types.ModuleType("google.adk.events")

_tools_base.BaseTool = _BaseTool
_tools_ctx.ToolContext = _ToolContext
_agents_base.BaseAgent = _BaseAgent
_agents_ctx.InvocationContext = _InvocationContext
_events_mod.Event = _Event

_google.adk = _adk
_adk.tools = _tools
_adk.agents = _agents
_adk.events = _events_mod
_tools.base_tool = _tools_base
_tools.tool_context = _tools_ctx
_agents.base_agent = _agents_base
_agents.invocation_context = _agents_ctx

sys.modules.setdefault("google", _google)
sys.modules.setdefault("google.adk", _adk)
sys.modules.setdefault("google.adk.tools", _tools)
sys.modules.setdefault("google.adk.tools.base_tool", _tools_base)
sys.modules.setdefault("google.adk.tools.tool_context", _tools_ctx)
sys.modules.setdefault("google.adk.agents", _agents)
sys.modules.setdefault("google.adk.agents.base_agent", _agents_base)
sys.modules.setdefault("google.adk.agents.invocation_context", _agents_ctx)
sys.modules.setdefault("google.adk.events", _events_mod)

# ---- Stub dependencies for a2a ----
class _FakeA2AClient:
    def __init__(self, *, httpx_client=None, url=None):
        self.httpx_client = httpx_client
        self.url = url
        self.sent_requests = []

    async def send_message(self, request):
        self.sent_requests.append(request)
        return types.SimpleNamespace(
            root=types.SimpleNamespace(
                result=types.SimpleNamespace(
                    parts=[types.SimpleNamespace(text="server reply")]
                )
            )
        )

def create_text_message_object(*, content="", role="user"):
    return types.SimpleNamespace(role=role, parts=[types.SimpleNamespace(text=content)])

class MessageSendParams:
    def __init__(self, *, message):
        self.message = message

class SendMessageRequest:
    def __init__(self, *, params):
        self.params = params

_a2a_client_mod = types.ModuleType("a2a.client")
_a2a_client_mod.A2AClient = _FakeA2AClient
_a2a_client_mod.create_text_message_object = create_text_message_object
_a2a_client_mod.MessageSendParams = MessageSendParams
_a2a_client_mod.SendMessageRequest = SendMessageRequest

sys.modules.setdefault("a2a.client", _a2a_client_mod)

# Import library after stubbing
from adk_a2a_lib import A2ATool, A2ARemoteAgent


@pytest.mark.asyncio
async def test_a2a_tool_run_async():
    client = _FakeA2AClient()
    tool = A2ATool(client=client)
    result = await tool.run_async(args={"task": "hello"}, tool_context=_ToolContext())
    assert result == "server reply"
    assert client.sent_requests[0].params.message.parts[0].text == "hello"


@pytest.mark.asyncio
async def test_a2a_tool_missing_input():
    tool = A2ATool(client=_FakeA2AClient())
    with pytest.raises(ValueError):
        await tool.run_async(args={}, tool_context=_ToolContext())


@pytest.mark.asyncio
async def test_a2a_tool_from_url():
    httpx_client = object()
    tool = await A2ATool.from_url(url="http://example.com", httpx_client=httpx_client)
    assert isinstance(tool, A2ATool)
    assert isinstance(tool._client, _FakeA2AClient)
    assert tool._client.httpx_client is httpx_client
    assert tool._client.url == "http://example.com"


@pytest.mark.asyncio
async def test_a2a_remote_agent_run_async_impl():
    client = _FakeA2AClient()
    agent = A2ARemoteAgent(client=client, name="remote")
    user_content = create_text_message_object(content="hi", role="user")
    ctx = _InvocationContext(invocation_id="123", branch="b1", user_content=user_content)
    events = [e async for e in agent._run_async_impl(ctx)]
    assert len(events) == 2
    assert events[0].author == "remote"
    assert events[1].content.parts[0].text == "server reply"
