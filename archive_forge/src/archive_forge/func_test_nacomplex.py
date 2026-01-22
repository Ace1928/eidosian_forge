import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
@pytest.mark.xfail(reason='Edge case with conversion.')
def test_nacomplex():
    vec = robjects.ComplexVector((1 + 1j, 2 + 2j, 3 + 3j))
    vec[0] = robjects.NA_Complex
    assert robjects.baseenv['is.na'](vec)[0] is True