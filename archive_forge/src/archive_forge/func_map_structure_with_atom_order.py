import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
def map_structure_with_atom_order(in_list: list, first_call: bool=True) -> list:
    if first_call:
        in_list = copy.deepcopy(in_list)
    for i in range(len(in_list)):
        if isinstance(in_list[i], list):
            in_list[i] = map_structure_with_atom_order(in_list[i], first_call=False)
        elif isinstance(in_list[i], str):
            in_list[i] = atom_order[in_list[i]]
        else:
            raise ValueError('Unexpected type when mapping nested lists!')
    return in_list