import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def wald_pdf(x):
    if x > 0:
        return math.exp(-(x - 1) ** 2 / (2 * x)) / math.sqrt(x ** 3)
    return 0.0