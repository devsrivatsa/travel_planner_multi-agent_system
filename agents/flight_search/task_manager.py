from common.server.task_manager import InMemoryTaskManager
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import logging
from common.types import (
    SendTaskRequest, 
    SendTaskResponse, 
    TaskSendParams, 
    Artifact, 
    TaskStatus, 
    Message, 
    TaskState,
    Task,
    TextPart,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    JSONRPCResponse
)
from typing import AsyncIterable
logger = logging.getLogger(__name__)

class FlightAgentTaskManager(InMemoryTaskManager):
    """Task manager specific to the flight search agent"""
    def __init__(self, agent:Agent, runner:Runner, session_service:InMemorySessionService):
        super().__init__()
        self.agent = agent
        self.runner = runner
        self.session_service = session_service
    
    async def _update_store(self, task_id: str, status:TaskStatus, artifacts:list[Artifact]) -> Task:
        async with self.lock:
            try:
                task = self.tasks[task_id]
            except KeyError:
                logger.error(f"Task {task_id} not found for updating the task")
                raise ValueError(f"Task {task_id} not found")
            task.status = status
            if status.message is not None:
                self.task_messages[task_id].append(status.message)
            if artifacts is not None:
                if task.artifacts is None:
                    task.artifacts = []
                task.artifacts.extend(artifacts)
            return task

    async def _invoke(self, request: SendTaskRequest) -> SendTaskResponse:
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        try:
            result = self.agent.invoke(query, task_send_params.sessionId)
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            raise ValueError(f"Error invoking agent: {e}")
        parts = [{"type": "tex", "text": result}]
        task_state = TaskState.INPUT_REQUIRED if "MISSING_INFO:" in result else TaskState.COMPLETED
        task = await self._update_store(
            task_send_params.id,
            TaskStatus(state=task_state, message=Message(role="agent", parts=parts)),
            [Artifact(parts=parts)]
        )
        return SendTaskResponse(id=request.id, result=task)
    
    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        part = task_send_params.messages.parts[0]
        if not isinstance(part, TextPart):
            raise ValueError("User query is not a text part. Only text parts are supported")
        return part.text
    
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        error = self._validate_request(request)
        if error:
            return error
        await self.upsert_task(request.params)
        return await self._invoke(request)
    
    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        error = self._validate_request(request)
        if error:
            return error
        await self.upsert_task(request.params)
        return self._stream_generator(request)
