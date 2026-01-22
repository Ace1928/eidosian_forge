from unittest import mock
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot, InterProjectionTransform
@pytest.mark.mpl_image_compare(filename='geoaxes_set_boundary_clipping.png')
def test_geoaxes_set_boundary_clipping():
    """Test that setting the boundary works properly for clipping #1620."""
    lon, lat = np.meshgrid(np.linspace(-180.0, 180.0, 361), np.linspace(-90.0, -60.0, 31))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax1.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
    ax1.gridlines()
    ax1.contourf(lon, lat, lat, transform=ccrs.PlateCarree())
    ax1.set_boundary(mpath.Path.circle(center=(0.5, 0.5), radius=0.5), transform=ax1.transAxes)
    return fig