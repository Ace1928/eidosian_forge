from __future__ import annotations
from pydantic import BaseModel as _BaseModel
from typing import TYPE_CHECKING, Optional
@classmethod
def get_flow_schema(cls, name: str) -> 'AZFlowSchema':
    """
        Returns the AZFlowSchema
        """
    from ..utils.lazy import get_az_flow_schema
    return get_az_flow_schema(name)