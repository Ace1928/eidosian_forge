import logging
import warnings
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_community.utils.openai import is_openai_v1
@property
def default_params(self) -> Dict[str, Any]:
    return {'model': self.model, 'temperature': self.temperature, 'top_p': self.top_p, 'top_k': self.top_k, 'max_tokens': self.max_tokens, 'repetition_penalty': self.repetition_penalty}