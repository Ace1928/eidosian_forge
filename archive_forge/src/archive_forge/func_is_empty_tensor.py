import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def is_empty_tensor(x):
    x_shape = x.meta['example_value'].shape
    return len(x_shape) == 1 and x_shape[0] == 0