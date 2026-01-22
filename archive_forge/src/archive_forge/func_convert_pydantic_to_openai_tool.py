from __future__ import annotations
import inspect
import uuid
from typing import (
from typing_extensions import TypedDict
from langchain_core._api import deprecated
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs
@deprecated('0.1.16', alternative='langchain_core.utils.function_calling.convert_to_openai_tool()', removal='0.2.0')
def convert_pydantic_to_openai_tool(model: Type[BaseModel], *, name: Optional[str]=None, description: Optional[str]=None) -> ToolDescription:
    """Converts a Pydantic model to a function description for the OpenAI API."""
    function = convert_pydantic_to_openai_function(model, name=name, description=description)
    return {'type': 'function', 'function': function}