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
def test_posixct_in_dataframe_to_pandas(self):
    tzone = robjects.vectors.get_timezone()
    dt = [datetime(1960, 5, 2), datetime(1970, 6, 3), datetime(2012, 7, 1)]
    dt = [x.replace(tzinfo=tzone) for x in dt]
    ts = [x.timestamp() for x in dt]
    r_dataf = robjects.vectors.DataFrame({'mydate': robjects.baseenv['as.POSIXct'](rinterface.FloatSexpVector(ts), origin=rinterface.StrSexpVector(('1970-01-01',)))})
    with localconverter(default_converter + rpyp.converter):
        py_dataf = robjects.conversion.converter_ctx.get().rpy2py(r_dataf)
    assert pandas.core.dtypes.common.is_datetime64_any_dtype(py_dataf['mydate'])