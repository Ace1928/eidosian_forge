from datetime import datetime
import pytest
from cartopy.feature.nightshade import Nightshade, _julian_day, _solar_position
@pytest.mark.parametrize('dt, true_lat, true_lon', [(datetime(2018, 9, 29, 0, 0), -(2 + 18 / 60), 177 + 37 / 60), (datetime(2018, 9, 29, 14, 0), -(2 + 32 / 60), -(32 + 25 / 60)), (datetime(1992, 2, 14, 0, 0), -(13 + 20 / 60), -(176 + 26 / 60)), (datetime(2030, 6, 21, 0, 0), 23 + 26 / 60, -(179 + 34 / 60))])
def test_solar_position(dt, true_lat, true_lon):
    lat, lon = _solar_position(dt)
    assert pytest.approx(true_lat, 0.1) == lat
    assert pytest.approx(true_lon, 0.1) == lon