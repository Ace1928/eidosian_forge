import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
def test_POSIXlt_from_invalidpythontime():
    x = [time.struct_time(_dateval_tuple), time.struct_time(_dateval_tuple)]
    x.append('foo')
    with pytest.raises(ValueError):
        robjects.POSIXlt(x)