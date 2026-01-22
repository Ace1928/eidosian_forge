import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def shape_env_check_state_equal(env1, env2, non_state_variable_names, map_value):
    env1_vars = vars(env1).copy()
    env2_vars = vars(env2).copy()
    for v in non_state_variable_names:
        if v in env1_vars:
            env1_vars.pop(v)
        if v in env2_vars:
            env2_vars.pop(v)

    def value_to_str(value: Any) -> str:
        if isinstance(value, dict):
            return '{' + ', '.join((f'{k}: {value[k]}' for k in sorted(value.keys(), key=str))) + '}'
        if isinstance(value, set):
            return '{' + ', '.join((f'{v}' for v in sorted(value))) + '}'
        return str(value)

    def compare_vars(map_value: Callable[[str, Any], Any]) -> List[Tuple[str, str, str]]:
        env1_set, env2_set = (set(env1_vars), set(env2_vars))
        if env1_set != env2_set:
            raise NotEqualError('field set mismatch:', [('found unique fields:', str(sorted(env1_set - env2_set)), str(sorted(env2_set - env1_set)))])
        sorted_keys = list(env1_set)
        sorted_keys.sort()
        mapped_dict = [(k, map_value(k, env1_vars[k]), map_value(k, env2_vars[k])) for k in sorted_keys]
        return [(f"{k}: values don't match.", value_to_str(val1), value_to_str(val2)) for k, val1, val2 in mapped_dict if val1 != val2]
    errors = compare_vars(map_value)
    if len(errors) > 0:
        raise NotEqualError("field values don't match:", errors)