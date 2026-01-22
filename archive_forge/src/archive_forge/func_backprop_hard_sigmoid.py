import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def backprop_hard_sigmoid(self, dY: FloatsXdT, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
    return self.backprop_clipped_linear(dY, X, slope=0.2, offset=0.5)