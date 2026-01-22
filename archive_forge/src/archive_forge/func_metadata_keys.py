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
def metadata_keys(self) -> List[str]:
    keys: List[str] = []
    if self.is_empty:
        return keys
    for field_name in self.__fields__.keys():
        field_group = getattr(self, field_name)
        if field_group is not None:
            for field in field_group:
                if not isinstance(field, str) and field.name not in [self.content_key, self.content_vector_key]:
                    keys.append(field.name)
    return keys