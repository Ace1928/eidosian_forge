import inspect
import pathlib
import sys
import typing
from collections import defaultdict
from types import CodeType
from typing import Dict, Iterable, List, Optional
import torch
def consolidate_types(self, qualified_name: str) -> Dict:
    all_args = self.analyze(qualified_name)
    for arg, types in all_args.items():
        types = list(types)
        type_length = len(types)
        if type_length == 2 and type(None) in types:
            all_args[arg] = get_optional_of_element_type(types)
        elif type_length > 1:
            all_args[arg] = 'Any'
        elif type_length == 1:
            all_args[arg] = get_type(types[0])
    return all_args