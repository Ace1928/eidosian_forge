import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@validator('top_k')
def valid_top_k(cls, v):
    if v is not None and v <= 0:
        raise ValueError('`top_k` must be strictly positive')
    return v