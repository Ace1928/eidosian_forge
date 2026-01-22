from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def entry_to_pretty_str(entry) -> str:
    """
    Given a backend_config_dict entry, returns a string with the human readable
    representation of it.
    """
    s = '{\n'
    if 'pattern' in entry:
        pattern_str = pattern_to_human_readable(entry['pattern'])
        s += f"  'pattern': {pattern_str},\n"
    if 'dtype_configs' in entry:
        s += "  'dtype_configs': [\n"
        for dtype_config in entry['dtype_configs']:
            s += '    {\n'
            for k, v in dtype_config.items():
                s += f"      '{k}': {v},\n"
            s += '    },\n'
        s += '  ],\n'
    if 'num_tensor_args_to_observation_type' in entry:
        s += "  'num_tensor_args_to_observation_type': {\n"
        for k, v in entry['num_tensor_args_to_observation_type'].items():
            s += f'    {k}: {v},\n'
        s += '  },\n'
    custom_handled_fields = ['pattern', 'dtype_configs', 'num_tensor_args_to_observation_type']
    for field_name in entry:
        if field_name in custom_handled_fields:
            continue
        s += f"  '{field_name}': {entry[field_name]},\n"
    s += '}'
    return s