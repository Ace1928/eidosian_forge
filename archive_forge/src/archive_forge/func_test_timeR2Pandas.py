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
def test_timeR2Pandas(self):
    tzone = robjects.vectors.get_timezone()
    dt = [datetime(1960, 5, 2), datetime(1970, 6, 3), datetime(2012, 7, 1)]
    dt = [x.replace(tzinfo=tzone) for x in dt]
    ts = [x.timestamp() for x in dt]
    r_time = robjects.baseenv['as.POSIXct'](rinterface.FloatSexpVector(ts), origin=rinterface.StrSexpVector(('1970-01-01',)))
    with localconverter(default_converter + rpyp.converter) as cv:
        py_time = robjects.conversion.converter_ctx.get().rpy2py(r_time)
    for expected, obtained in zip(dt, py_time):
        assert expected == obtained.to_pydatetime()
    r_time[1] = rinterface.na_values.NA_Real
    with localconverter(default_converter + rpyp.converter) as cv:
        py_time = robjects.conversion.converter_ctx.get().rpy2py(r_time)
    assert py_time[1] is pandas.NaT