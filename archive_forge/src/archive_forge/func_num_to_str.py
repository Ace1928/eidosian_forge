import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
def num_to_str(self, value: Optional[float], estimated_sigfigs: int, spread: Optional[float]):
    if value is None:
        return ' ' * len(self.num_to_str(1, estimated_sigfigs, None))
    if self._trim_significant_figures:
        value = common.trim_sigfig(value, estimated_sigfigs)
    return self._template.format(value, f' (! {spread * 100:.0f}%)' if self._highlight_warnings and spread is not None else '')