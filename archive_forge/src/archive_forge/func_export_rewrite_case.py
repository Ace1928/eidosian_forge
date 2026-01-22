import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
def export_rewrite_case(**kwargs):

    def wrapper(m):
        configs = kwargs
        parent = configs.pop('parent')
        assert isinstance(parent, ExportCase)
        key = parent.name
        if key not in _EXAMPLE_REWRITE_CASES:
            _EXAMPLE_REWRITE_CASES[key] = []
        configs['example_inputs'] = parent.example_inputs
        case = _make_export_case(m, to_snake_case(m.__name__), configs)
        _EXAMPLE_REWRITE_CASES[key].append(case)
        return case
    return wrapper