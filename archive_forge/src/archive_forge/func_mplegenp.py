import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
def mplegenp(nu, mu, x):
    if mu == int(mu) and x == 1:
        if mu == 0:
            return 1
        else:
            return 0
    return mpmath.legenp(nu, mu, x)