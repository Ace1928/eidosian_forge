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
def parse_tool_calls(raw_tool_calls: List[dict], *, partial: bool=False, strict: bool=False, return_id: bool=True) -> List[Dict[str, Any]]:
    """Parse a list of tool calls."""
    final_tools: List[Dict[str, Any]] = []
    exceptions = []
    for tool_call in raw_tool_calls:
        try:
            parsed = parse_tool_call(tool_call, partial=partial, strict=strict, return_id=return_id)
            if parsed:
                final_tools.append(parsed)
        except OutputParserException as e:
            exceptions.append(str(e))
            continue
    if exceptions:
        raise OutputParserException('\n\n'.join(exceptions))
    return final_tools