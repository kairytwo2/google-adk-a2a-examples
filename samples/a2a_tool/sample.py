import asyncio
from adk_a2a_lib import A2ATool
from google.adk.tools.tool_context import ToolContext

async def main() -> None:
    tool = await A2ATool.from_url(url="http://localhost:8000/")
    result = await tool.run_async(args={"task": "demo"}, tool_context=ToolContext())
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
