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
def test_series_obj_allnone(self):
    Series = pandas.core.series.Series
    s = Series([None, None, None], index=['a', 'b', 'c'])
    with localconverter(default_converter + rpyp.converter) as cv:
        rp_s = robjects.conversion.converter_ctx.get().py2rpy(s)
    assert isinstance(rp_s, rinterface.BoolSexpVector)