import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
@pytest.mark.parametrize('x', ([time.struct_time(_dateval_tuple)] * 2, [datetime.datetime(*_dateval_tuple[:-2])] * 2))
def test_POSIXct_from_python_times(x, default_timezone_mocker):
    res = robjects.POSIXct(x)
    assert list(res.slots['class']) == ['POSIXct', 'POSIXt']
    assert len(res) == 2
    zone = default_timezone_mocker
    assert res.slots['tzone'][0] == (zone if zone else '')