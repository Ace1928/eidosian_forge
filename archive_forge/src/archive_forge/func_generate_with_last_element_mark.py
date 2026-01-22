from __future__ import annotations
import asyncio
import functools
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
def generate_with_last_element_mark(iterable: Iterable[T]) -> Iterator[Tuple[T, bool]]:
    """Generate elements from an iterable,
    and a boolean indicating if it is the last element."""
    iterator = iter(iterable)
    try:
        item = next(iterator)
    except StopIteration:
        return
    for next_item in iterator:
        yield (item, False)
        item = next_item
    yield (item, True)