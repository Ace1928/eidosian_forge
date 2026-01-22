import re
import warnings
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
from langchain_core.utils.utils import build_extra_kwargs, convert_to_secret_str
def convert_prompt(self, prompt: PromptValue) -> str:
    return self._wrap_prompt(prompt.to_string())