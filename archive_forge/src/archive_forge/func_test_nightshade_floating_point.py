from datetime import datetime
import pytest
from cartopy.feature.nightshade import Nightshade, _julian_day, _solar_position
def test_nightshade_floating_point():
    date = datetime(1999, 12, 31, 12)
    Nightshade(date, refraction=-6.0, color='none')