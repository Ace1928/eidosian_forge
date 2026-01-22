from __future__ import annotations
import logging
import os
import sys
import warnings
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from langchain_core.utils.utils import build_extra_kwargs
from langchain_community.utils.openai import is_openai_v1
def get_sub_prompts(self, params: Dict[str, Any], prompts: List[str], stop: Optional[List[str]]=None) -> List[List[str]]:
    """Get the sub prompts for llm call."""
    if stop is not None:
        if 'stop' in params:
            raise ValueError('`stop` found in both the input and default params.')
        params['stop'] = stop
    if params['max_tokens'] == -1:
        if len(prompts) != 1:
            raise ValueError('max_tokens set to -1 not supported for multiple inputs.')
        params['max_tokens'] = self.max_tokens_for_prompt(prompts[0])
    sub_prompts = [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]
    return sub_prompts