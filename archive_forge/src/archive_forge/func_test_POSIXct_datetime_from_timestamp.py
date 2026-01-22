import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
def test_POSIXct_datetime_from_timestamp(default_timezone_mocker):
    tzone = robjects.vectors.get_timezone()
    dt = [datetime.datetime(1900, 1, 1), datetime.datetime(1970, 1, 1), datetime.datetime(2000, 1, 1)]
    dt = [x.replace(tzinfo=tzone) for x in dt]
    ts = [x.timestamp() for x in dt]
    res = [robjects.POSIXct._datetime_from_timestamp(x, tzone) for x in ts]
    for expected, obtained in zip(dt, res):
        assert expected == obtained