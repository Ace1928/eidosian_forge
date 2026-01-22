import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_36
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='igh_land.png', tolerance=0.5 if _MPL_36 else 3.6)
def test_igh_land():
    crs = ccrs.InterruptedGoodeHomolosine(emphasis='land')
    ax = plt.axes(projection=crs)
    ax.coastlines()
    ax.gridlines()
    return ax.figure