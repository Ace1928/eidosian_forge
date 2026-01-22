import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest
from scipy.special import _ufuncs
import scipy.special._orthogonal as orth
from scipy.special._testutils import FuncData

    Check that the eval_* functions sig='ld->d' and 'dd->d' agree.

    