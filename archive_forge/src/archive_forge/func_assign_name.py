from __future__ import annotations
import datetime
import warnings
from typing import Any, Dict, List, Optional, Type
from uuid import UUID
from langsmith.schemas import RunBase as BaseRunV2
from langsmith.schemas import RunTypeEnum as RunTypeEnumDep
from langchain_core._api import deprecated
from langchain_core.outputs import LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
@root_validator(pre=True)
def assign_name(cls, values: dict) -> dict:
    """Assign name to the run."""
    if values.get('name') is None:
        if 'name' in values['serialized']:
            values['name'] = values['serialized']['name']
        elif 'id' in values['serialized']:
            values['name'] = values['serialized']['id'][-1]
    if values.get('events') is None:
        values['events'] = []
    return values