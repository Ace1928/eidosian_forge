import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
def optional_min(seq):
    l = list(seq)
    return None if len(l) == 0 else min(l)