from __future__ import annotations
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
def chat_with_retry(llm: ChatPremAI, project_id: int, messages: List[dict], stream: bool=False, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Any:
    """Using tenacity for retry in completion call"""
    retry_decorator = create_prem_retry_decorator(llm, max_retries=llm.max_retries, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(project_id: int, messages: List[dict], stream: Optional[bool]=False, **kwargs: Any) -> Any:
        response = llm.client.chat.completions.create(project_id=project_id, messages=messages, stream=stream, **kwargs)
        return response
    return _completion_with_retry(project_id=project_id, messages=messages, stream=stream, **kwargs)