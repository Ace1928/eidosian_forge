import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def dsigmoid(self, Y: FloatsXdT, *, inplace: bool=False) -> FloatsXdT:
    if inplace:
        Y *= 1 - Y
        return Y
    else:
        return Y * (1.0 - Y)