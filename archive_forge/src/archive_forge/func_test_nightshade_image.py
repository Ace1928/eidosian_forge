from datetime import datetime
import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='nightshade_platecarree.png')
def test_nightshade_image():
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    dt = datetime(2018, 11, 10, 0, 0)
    ax.set_global()
    ax.add_feature(Nightshade(dt, alpha=0.75))
    return ax.figure