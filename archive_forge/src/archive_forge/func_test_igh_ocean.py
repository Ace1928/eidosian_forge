import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_36
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='igh_ocean.png', tolerance=0.5 if _MPL_36 else 4.5)
def test_igh_ocean():
    crs = ccrs.InterruptedGoodeHomolosine(central_longitude=-160, emphasis='ocean')
    ax = plt.axes(projection=crs)
    ax.coastlines()
    ax.gridlines()
    return ax.figure