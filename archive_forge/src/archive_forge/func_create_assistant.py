from __future__ import annotations
import json
from json import JSONDecodeError
from time import sleep
from typing import (
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager
from langchain_core.load import dumpd
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
@classmethod
def create_assistant(cls, name: str, instructions: str, tools: Sequence[Union[BaseTool, dict]], model: str, *, client: Optional[Union[openai.OpenAI, openai.AzureOpenAI]]=None, **kwargs: Any) -> OpenAIAssistantRunnable:
    """Create an OpenAI Assistant and instantiate the Runnable.

        Args:
            name: Assistant name.
            instructions: Assistant instructions.
            tools: Assistant tools. Can be passed in OpenAI format or as BaseTools.
            model: Assistant model to use.
            client: OpenAI or AzureOpenAI client.
                Will create default OpenAI client if not specified.

        Returns:
            OpenAIAssistantRunnable configured to run using the created assistant.
        """
    client = client or _get_openai_client()
    assistant = client.beta.assistants.create(name=name, instructions=instructions, tools=[_get_assistants_tool(tool) for tool in tools], model=model, file_ids=kwargs.get('file_ids'))
    return cls(assistant_id=assistant.id, client=client, **kwargs)