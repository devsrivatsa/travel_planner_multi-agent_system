from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from agent import get_agent_async
from task_manager import FlightAgentTaskManager
from dotenv import load_dotenv
import logging
import uvicorn
import os

load_dotenv()

logger = logging.getLogger(__name__)

async def main(host, port):
    try:
        if not os.getenv("GEMINI_API_KEY"):
            raise MissingAPIKeyError("GEMINI_API_KEY is not set")
        
        capabilities = AgentCapabilities(streaming=True)
        flight_search_skill = AgentSkill(
            id="search_flights",
            name="Search Flights",
            description="Searches for flights based on origin, destination, and date",
            tags=["flights", "travel"],
            examples=["find flights from JFK to LAX tomorrow"]
        )
        agent_card = AgentCard(
            name = "Flight Search Agent",
            description = "An agent that searches for flights based on origin, destination, and date",
            skills = [flight_search_skill],
            capabilities = capabilities,
            url = f"http://{host}:{port}",
            version = "1.0.0",
            defaultInputModes = ["text"],
            defaultOutputModes = ["text"],
        )
        session_service = InMemorySessionService()
        agent, exit_stack = await get_agent_async()
        runner = Runner(
            app_name = "flight_search_a2a_app",
            agent = agent,
            session_service = session_service
        )
        task_manager = FlightAgentTaskManager(
            agent=agent,
            runner=runner,
            session_service=session_service
        )
        a2a_server = A2AServer(
            agent_card = agent_card,
            task_manager = task_manager,
            host = host,
            port = port
        )
        config = uvicorn.Config(
            app = a2a_server.app,
            host = host,
            port = port,
            log_level = "info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    except Exception as e:
        logger.error(f"Error starting flight search A2A server: {e}")
        exit(1)
    finally:
        if exit_stack:
            await exit_stack.aclose()


if __name__ == "__main__":
    import asyncio
    host = "0.0.0.0"
    port = 8000
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(main(host, port))
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        loop.close()
