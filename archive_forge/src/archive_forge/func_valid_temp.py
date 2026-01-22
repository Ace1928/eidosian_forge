import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@validator('temperature')
def valid_temp(cls, v):
    if v is not None and v <= 0:
        raise ValueError('`temperature` must be strictly positive')
    return v