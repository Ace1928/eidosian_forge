import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
@pytest.fixture(scope='module', params=_zones_str)
def default_timezone_mocker(request):
    zone_str = request.param
    if zone_str:
        rpy2.robjects.vectors.default_timezone = zoneinfo.ZoneInfo(zone_str)
    yield zone_str
    rpy2.robjects.vectors.default_timezone = None