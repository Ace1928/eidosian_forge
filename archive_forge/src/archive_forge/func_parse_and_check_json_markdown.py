from __future__ import annotations
import json
import re
from typing import Any, Callable, List
from langchain_core.exceptions import OutputParserException
def parse_and_check_json_markdown(text: str, expected_keys: List[str]) -> dict:
    """
    Parse a JSON string from a Markdown string and check that it
    contains the expected keys.

    Args:
        text: The Markdown string.
        expected_keys: The expected keys in the JSON string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    try:
        json_obj = parse_json_markdown(text)
    except json.JSONDecodeError as e:
        raise OutputParserException(f'Got invalid JSON object. Error: {e}')
    for key in expected_keys:
        if key not in json_obj:
            raise OutputParserException(f'Got invalid return object. Expected key `{key}` to be present, but got {json_obj}')
    return json_obj