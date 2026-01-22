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
@pytest.mark.parametrize('rcode,names,values', (('list(list(x=1, y=3), list(x=2, y=4))', ('x', 'y'), ((1, 2), (3, 4))), ('list(list(x=1), list(x=2, y=4))', ('x', 'y'), ((1, 2), (None, 4)))))
def test__records_to_columns(self, rcode, names, values):
    rlist = robjects.r(rcode)
    columns = rpyp._records_to_columns(rlist)
    assert tuple(columns.keys()) == names
    for n, v in zip(names, values):
        assert tuple(columns[n]) == v