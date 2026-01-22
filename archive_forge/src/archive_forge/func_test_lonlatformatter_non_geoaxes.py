from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
@pytest.mark.parametrize('cls,letter', [(LongitudeFormatter, 'E'), (LatitudeFormatter, 'N')])
def test_lonlatformatter_non_geoaxes(cls, letter):
    ticks = [2, 2.5]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, 10], [0, 1])
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(cls(degree_symbol='', dms=False))
    fig.canvas.draw()
    ticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert ticklabels == [f'{v:g}{letter}' for v in ticks]