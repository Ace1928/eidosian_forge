from __future__ import annotations
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import (
from typing_extensions import TypeAlias
from langchain_core._api import beta, deprecated
from langchain_core.messages import (
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.runnables import Runnable, RunnableSerializable
from langchain_core.utils import get_pydantic_field_names
@beta()
def with_structured_output(self, schema: Union[Dict, Type[BaseModel]], **kwargs: Any) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
    """Implement this if there is a way of steering the model to generate responses that match a given schema."""
    raise NotImplementedError()