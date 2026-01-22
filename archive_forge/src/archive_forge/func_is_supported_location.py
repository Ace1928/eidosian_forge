from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@staticmethod
def is_supported_location(location: str) -> bool:
    """Return whether the provided location is supported."""
    try:
        return APIPropertyLocation.from_str(location) in SUPPORTED_LOCATIONS
    except ValueError:
        return False