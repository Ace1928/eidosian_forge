import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
@validator('top_k')
def validate_top_k(cls, value: int) -> int:
    if value < 0:
        raise ValueError(f'top_k ({value}) cannot be negative.')
    return value