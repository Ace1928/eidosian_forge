import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='xticks_cylindrical.png')
def test_set_xticks_cylindrical():
    ax = plt.axes(projection=ccrs.Mercator(min_latitude=-85, max_latitude=85))
    ax.coastlines('110m')
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
    ax.set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())
    ax.set_xticks([-135, -45, 45, 135], minor=True, crs=ccrs.PlateCarree())
    return ax.figure