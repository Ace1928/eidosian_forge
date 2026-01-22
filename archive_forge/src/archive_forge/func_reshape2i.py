import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def reshape2i(self, array: IntsXd, d0: int, d1: int) -> Ints2d:
    return cast(Ints2d, self.reshape(array, (d0, d1)))