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
def test_series_issue264(self):
    Series = pandas.core.series.Series
    s = Series(('a', 'b', 'c', 'd', 'e'), index=pandas.Index([0, 1, 2, 3, 4], dtype='int64'))
    with localconverter(default_converter + rpyp.converter) as cv:
        rp_s = robjects.conversion.converter_ctx.get().py2rpy(s)
    str(rp_s)
    assert isinstance(rp_s, rinterface.StrSexpVector)