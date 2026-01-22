import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@validator('top_p')
def valid_top_p(cls, v):
    if v is not None and (v <= 0 or v >= 1.0):
        raise ValueError('`top_p` must be > 0.0 and < 1.0')
    return v