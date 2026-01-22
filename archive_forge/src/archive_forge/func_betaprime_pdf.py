import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def betaprime_pdf(x, a, b):
    if x > 0:
        logf = (a - 1) * math.log(x) - (a + b) * math.log1p(x) - sc.betaln(a, b)
        return math.exp(logf)
    elif a > 1:
        return 0
    elif a < 1:
        return np.inf
    else:
        return 1 / sc.beta(a, b)