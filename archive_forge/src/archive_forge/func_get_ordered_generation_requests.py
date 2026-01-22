from __future__ import annotations
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.adapters.openai import (
from langchain_community.chat_models.openai import _convert_delta_to_message_chunk
def get_ordered_generation_requests(models_priority_list: List[GPTRouterModel], **kwargs: Any) -> List:
    """
    Return the body for the model router input.
    """
    from gpt_router.models import GenerationParams, ModelGenerationRequest
    return [ModelGenerationRequest(model_name=model.name, provider_name=model.provider_name, order=index + 1, prompt_params=GenerationParams(**kwargs)) for index, model in enumerate(models_priority_list)]