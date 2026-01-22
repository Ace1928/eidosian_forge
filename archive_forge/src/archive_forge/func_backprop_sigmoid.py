import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def backprop_sigmoid(self, dY: FloatsXdT, Y: FloatsXdT, *, inplace: bool=False) -> FloatsXdT:
    if inplace:
        self.dsigmoid(Y, inplace=True)
        Y *= dY
        return Y
    else:
        return dY * self.dsigmoid(Y, inplace=inplace)