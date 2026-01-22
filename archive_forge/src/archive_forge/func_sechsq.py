import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def sechsq(self, X: FloatsXdT) -> FloatsXdT:
    X = self.xp.clip(X, -20.0, 20.0)
    return (1 / self.xp.cosh(X)) ** 2