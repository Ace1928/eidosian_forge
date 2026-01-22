import ast
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict
def evaluate_assign(assign, state, tools):
    var_names = assign.targets
    result = evaluate_ast(assign.value, state, tools)
    if len(var_names) == 1:
        state[var_names[0].id] = result
    else:
        if len(result) != len(var_names):
            raise InterpretorError(f'Expected {len(var_names)} values but got {len(result)}.')
        for var_name, r in zip(var_names, result):
            state[var_name.id] = r
    return result