import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
@pytest.mark.parametrize('zone_str', _zones_str[1:])
def test_POSIXct_from_python_timezone(zone_str):
    x = [datetime.datetime(*_dateval_tuple[:-2]).replace(tzinfo=zoneinfo.ZoneInfo(zone_str))] * 2
    res = robjects.POSIXct(x)
    assert list(res.slots['class']) == ['POSIXct', 'POSIXt']
    assert len(res) == 2
    assert res.slots['tzone'][0] == (zone_str if zone_str else '')