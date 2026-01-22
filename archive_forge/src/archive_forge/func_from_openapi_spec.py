from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@classmethod
def from_openapi_spec(cls, spec: OpenAPISpec, path: str, method: str) -> 'APIOperation':
    """Create an APIOperation from an OpenAPI spec."""
    operation = spec.get_operation(path, method)
    parameters = spec.get_parameters_for_operation(operation)
    properties = cls._get_properties_from_parameters(parameters, spec)
    operation_id = OpenAPISpec.get_cleaned_operation_id(operation, path, method)
    request_body = spec.get_request_body_for_operation(operation)
    api_request_body = APIRequestBody.from_request_body(request_body, spec) if request_body is not None else None
    description = operation.description or operation.summary
    if not description and spec.paths is not None:
        description = spec.paths[path].description or spec.paths[path].summary
    return cls(operation_id=operation_id, description=description or '', base_url=spec.base_url, path=path, method=method, properties=properties, request_body=api_request_body)