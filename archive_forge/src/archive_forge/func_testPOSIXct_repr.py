import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
def testPOSIXct_repr():
    sexp = robjects.r('ISOdate(2013, 12, 11)')
    res = robjects.POSIXct(sexp)
    s = repr(res)
    assert s.endswith('[2013-12-1...]')