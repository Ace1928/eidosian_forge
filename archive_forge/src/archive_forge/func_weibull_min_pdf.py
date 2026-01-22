import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def weibull_min_pdf(x, c):
    if x > 0:
        return c * math.exp((c - 1) * math.log(x) - x ** c)
    return 0.0