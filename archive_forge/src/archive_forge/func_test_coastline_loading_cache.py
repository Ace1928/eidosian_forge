import gc
from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.shapereader
from cartopy.mpl import _MPL_38
from cartopy.mpl.feature_artist import FeatureArtist
import cartopy.mpl.geoaxes as cgeoaxes
import cartopy.mpl.patch
@pytest.mark.natural_earth
def test_coastline_loading_cache():
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax1.coastlines()
    fig.canvas.draw()
    with mock.patch('cartopy.io.shapereader.Reader.__init__', return_value=None) as counter:
        ax2 = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson())
        ax2.coastlines()
        fig.canvas.draw()
    counter.assert_not_called()