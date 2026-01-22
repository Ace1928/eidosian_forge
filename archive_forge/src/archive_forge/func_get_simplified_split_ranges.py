import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def get_simplified_split_ranges(self, split_sections: List[int], next_users: List[torch.fx.Node], user_inputs_list: List[List[Union[torch.fx.Node, _Range]]]) -> Optional[List[_Range]]:
    simplified_split_ranges = super().get_simplified_split_ranges(split_sections, next_users, user_inputs_list)
    if not simplified_split_ranges or len(simplified_split_ranges) != 1:
        return None
    return simplified_split_ranges