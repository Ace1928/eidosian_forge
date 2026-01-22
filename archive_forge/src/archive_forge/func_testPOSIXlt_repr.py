import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
def testPOSIXlt_repr():
    x = [time.struct_time(_dateval_tuple), time.struct_time(_dateval_tuple)]
    res = robjects.POSIXlt(x)
    s = repr(res)
    assert isinstance(s, str)