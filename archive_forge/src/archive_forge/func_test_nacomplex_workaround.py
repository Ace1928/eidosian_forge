import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_nacomplex_workaround():
    vec = robjects.ComplexVector((1 + 1j, 2 + 2j, 3 + 3j))
    vec[0] = complex(robjects.NA_Complex.real, robjects.NA_Complex.imag)
    assert robjects.baseenv['is.na'](vec)[0] is True