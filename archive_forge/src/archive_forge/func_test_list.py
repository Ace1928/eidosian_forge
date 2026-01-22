import math
from collections import OrderedDict
from datetime import datetime
import pytest
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import vectors
from rpy2.robjects import conversion
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
@pytest.mark.parametrize('cls', (robjects.ListVector, rinterface.ListSexpVector))
def test_list(self, cls):
    rlist = cls(robjects.ListVector({'a': 1, 'b': 'c'}))
    with localconverter(default_converter + rpyp.converter) as cv:
        pylist = cv.rpy2py(rlist)
    assert len(pylist) == 2
    assert set(pylist.keys()) == set(rlist.names)