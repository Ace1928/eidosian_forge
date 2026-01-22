import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def hard_swish_mobilenet(self, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
    if inplace:
        X *= self.relu_k(X + 3) / 6
        return X
    return X * (self.relu_k(X + 3) / 6)