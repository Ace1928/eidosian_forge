import logging
import warnings
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_community.utils.openai import is_openai_v1
def construct_payload(self, prompt: str, stop: Optional[List[str]]=None, **kwargs: Any) -> Dict[str, Any]:
    stop_to_use = stop[0] if stop and len(stop) == 1 else stop
    payload: Dict[str, Any] = {**self.default_params, 'prompt': prompt, 'stop': stop_to_use, **kwargs}
    return {k: v for k, v in payload.items() if v is not None}