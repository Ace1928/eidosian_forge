import base64
from abc import ABC
from datetime import datetime
from typing import Callable, Dict, Iterator, List, Literal, Optional, Union
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
@validator('since', allow_reuse=True)
def validate_since(cls, v: Optional[str]) -> Optional[str]:
    if v:
        try:
            datetime.strptime(v, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            raise ValueError(f"Invalid value for 'since'. Expected a date string in YYYY-MM-DDTHH:MM:SSZ format. Received: {v}")
    return v