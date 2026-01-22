import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def reshape2(self, array: ArrayXd, d0: int, d1: int) -> Array2d:
    return cast(Array2d, self.reshape(array, (d0, d1)))