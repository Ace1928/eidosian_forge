import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
@pytest.mark.xfail(reason='wday mismatch between R and Python (issue #523)')
def test_POSIXlt_getitem():
    x = [time.struct_time(_dateval_tuple), time.struct_time(_dateval_tuple)]
    res = robjects.POSIXlt(x)
    assert res[0] == x[0]