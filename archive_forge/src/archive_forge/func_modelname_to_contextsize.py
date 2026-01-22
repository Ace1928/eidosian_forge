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
@staticmethod
def modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model.

        Args:
            modelname: The modelname we want to know the context size for.

        Returns:
            The maximum context size

        Example:
            .. code-block:: python

                max_tokens = openai.modelname_to_contextsize("gpt-3.5-turbo-instruct")
        """
    model_token_mapping = {'gpt-4': 8192, 'gpt-4-0314': 8192, 'gpt-4-0613': 8192, 'gpt-4-32k': 32768, 'gpt-4-32k-0314': 32768, 'gpt-4-32k-0613': 32768, 'gpt-3.5-turbo': 4096, 'gpt-3.5-turbo-0301': 4096, 'gpt-3.5-turbo-0613': 4096, 'gpt-3.5-turbo-16k': 16385, 'gpt-3.5-turbo-16k-0613': 16385, 'gpt-3.5-turbo-instruct': 4096, 'text-ada-001': 2049, 'ada': 2049, 'text-babbage-001': 2040, 'babbage': 2049, 'text-curie-001': 2049, 'curie': 2049, 'davinci': 2049, 'text-davinci-003': 4097, 'text-davinci-002': 4097, 'code-davinci-002': 8001, 'code-davinci-001': 8001, 'code-cushman-002': 2048, 'code-cushman-001': 2048}
    if 'ft-' in modelname:
        modelname = modelname.split(':')[0]
    context_size = model_token_mapping.get(modelname, None)
    if context_size is None:
        raise ValueError(f'Unknown model: {modelname}. Please provide a valid OpenAI model name.Known models are: ' + ', '.join(model_token_mapping.keys()))
    return context_size