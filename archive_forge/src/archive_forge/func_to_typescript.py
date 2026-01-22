from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
def to_typescript(self) -> str:
    """Get typescript string representation of the operation."""
    operation_name = self.operation_id
    params = []
    if self.request_body:
        formatted_request_body_props = self._format_nested_properties(self.request_body.properties)
        params.append(formatted_request_body_props)
    for prop in self.properties:
        prop_name = prop.name
        prop_type = self.ts_type_from_python(prop.type)
        prop_required = '' if prop.required else '?'
        prop_desc = f'/* {prop.description} */' if prop.description else ''
        params.append(f'{prop_desc}\n\t\t{prop_name}{prop_required}: {prop_type},')
    formatted_params = '\n'.join(params).strip()
    description_str = f'/* {self.description} */' if self.description else ''
    typescript_definition = f'\n{description_str}\ntype {operation_name} = (_: {{\n{formatted_params}\n}}) => any;\n'
    return typescript_definition.strip()