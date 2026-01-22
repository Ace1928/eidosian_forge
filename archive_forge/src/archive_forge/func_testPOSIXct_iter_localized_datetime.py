import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
def testPOSIXct_iter_localized_datetime():
    timestamp = 1234567890
    timezone = 'UTC'
    r_vec = robjects.r('as.POSIXct')(timestamp, origin='1960-01-01', tz=timezone)
    r_times = robjects.r('strftime')(r_vec, format='%H:%M:%S', tz=timezone)
    py_value = next(r_vec.iter_localized_datetime())
    assert r_times[0] == ':'.join(('%i' % getattr(py_value, x) for x in ('hour', 'minute', 'second')))