import copy
import json
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Type
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, InvalidToolCall
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import BaseModel, ValidationError
from langchain_core.utils.json import parse_partial_json
def make_invalid_tool_call(raw_tool_call: Dict[str, Any], error_msg: Optional[str]) -> InvalidToolCall:
    """Create an InvalidToolCall from a raw tool call."""
    return InvalidToolCall(name=raw_tool_call['function']['name'], args=raw_tool_call['function']['arguments'], id=raw_tool_call.get('id'), error=error_msg)