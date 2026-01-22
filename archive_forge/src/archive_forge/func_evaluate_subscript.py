import ast
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict
def evaluate_subscript(subscript, state, tools):
    index = evaluate_ast(subscript.slice, state, tools)
    value = evaluate_ast(subscript.value, state, tools)
    if isinstance(value, (list, tuple)):
        return value[int(index)]
    if index in value:
        return value[index]
    if isinstance(index, str) and isinstance(value, Mapping):
        close_matches = difflib.get_close_matches(index, list(value.keys()))
        if len(close_matches) > 0:
            return value[close_matches[0]]
    raise InterpretorError(f"Could not index {value} with '{index}'.")