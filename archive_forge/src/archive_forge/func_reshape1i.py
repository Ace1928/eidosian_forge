import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def reshape1i(self, array: IntsXd, d0: int) -> Ints1d:
    return cast(Ints1d, self.reshape(array, (d0,)))