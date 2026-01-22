import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str
from langchain_core.utils.env import get_from_dict_or_env
def get_batch_prompts(self, prompts: List[str]) -> List[List[str]]:
    """Get the sub prompts for llm call."""
    sub_prompts = [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]
    return sub_prompts