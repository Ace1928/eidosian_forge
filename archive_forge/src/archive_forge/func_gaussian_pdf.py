import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def gaussian_pdf(ops: Ops, X: FloatsXdT) -> FloatsXdT:
    """Gaussian PDF for distribution with mean 0 and stdev 1."""
    return INV_SQRT_2PI * ops.xp.exp(-0.5 * X * X)