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
def test_factorwithNA2Category(self):
    factor = robjects.vectors.FactorVector(('a', 'b', 'a', None))
    assert factor[3] is rinterface.na_values.NA_Integer
    with localconverter(default_converter + rpyp.converter) as cv:
        rp_c = robjects.conversion.converter_ctx.get().rpy2py(factor)
    assert isinstance(rp_c, pandas.Categorical)
    assert math.isnan(rp_c[3])