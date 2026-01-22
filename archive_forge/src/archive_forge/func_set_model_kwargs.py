import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
@validator('model_kwargs', always=True)
def set_model_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if v:
        assert 'prompt' not in v, "model_kwargs must not contain key 'prompt'"
        assert 'stop' not in v, "model_kwargs must not contain key 'stop'"
    return v