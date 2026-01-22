import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def gamma_pdf(x, a):
    if x > 0:
        return math.exp(-math.lgamma(a) + (a - 1.0) * math.log(x) - x)
    else:
        return 0 if a >= 1 else np.inf