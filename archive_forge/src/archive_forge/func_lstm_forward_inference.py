import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def lstm_forward_inference(self, params: Floats1d, H0: Floats3d, C0: Floats3d, X: Floats2d, size_at_t: Ints1d) -> Floats2d:
    Y, _ = lstm_forward_training(params, H0, C0, X, size_at_t)
    return Y