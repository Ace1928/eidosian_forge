import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
def test_pcolormesh_nan_wrap():
    xs, ys = np.meshgrid([120, 160, 200], [-30, 0, 30])
    data = np.ones((2, 2)) * np.nan
    ax = plt.axes(projection=ccrs.PlateCarree())
    mesh = ax.pcolormesh(xs, ys, data)
    pcolor = getattr(mesh, '_wrapped_collection_fix')
    if not _MPL_38:
        assert len(pcolor.get_paths()) == 2
    else:
        assert not pcolor.get_paths()
    mesh.set_array(np.ones((2, 2)))
    assert len(pcolor.get_paths()) == 2