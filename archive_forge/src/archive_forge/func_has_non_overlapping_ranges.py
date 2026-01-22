import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def has_non_overlapping_ranges(self, ranges: List[_Range]) -> bool:
    for range_, next_range in zip(ranges, ranges[1:]):
        if range_[1] > next_range[0]:
            return False
    return True