from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@classmethod
def from_openapi_url(cls, spec_url: str, path: str, method: str) -> 'APIOperation':
    """Create an APIOperation from an OpenAPI URL."""
    spec = OpenAPISpec.from_url(spec_url)
    return cls.from_openapi_spec(spec, path, method)