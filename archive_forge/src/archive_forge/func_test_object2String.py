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
def test_object2String(self):
    series = pandas.Series(['a', 'b', 'c', 'a'], dtype='O')
    with localconverter(default_converter + rpyp.converter) as cv:
        rp_c = robjects.conversion.converter_ctx.get().py2rpy(series)
        assert isinstance(rp_c, rinterface.StrSexpVector)