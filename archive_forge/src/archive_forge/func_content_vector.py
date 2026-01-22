from __future__ import annotations
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing_extensions import TYPE_CHECKING, Literal
from langchain_community.vectorstores.redis.constants import REDIS_VECTOR_DTYPE_MAP
@property
def content_vector(self) -> Union[FlatVectorField, HNSWVectorField]:
    if not self.vector:
        raise ValueError('No vector fields found')
    for field in self.vector:
        if field.name == self.content_vector_key:
            return field
    raise ValueError('No content_vector field found')