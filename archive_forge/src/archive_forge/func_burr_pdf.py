import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def burr_pdf(x, cc, dd):
    if x > 0:
        lx = math.log(x)
        return np.exp(-(cc + 1) * lx - (dd + 1) * math.log1p(np.exp(-cc * lx)))
    else:
        return 0