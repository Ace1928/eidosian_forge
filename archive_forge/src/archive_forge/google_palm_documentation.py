from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms import BaseLLM
from langchain_community.utilities.vertexai import create_retry_decorator
Get the number of tokens present in the text.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        