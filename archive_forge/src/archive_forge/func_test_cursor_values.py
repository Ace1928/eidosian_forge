import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
def test_cursor_values():
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    x, y = np.array([-969100.0, -4457000.0])
    r = ax.format_coord(x, y)
    assert r.encode('ascii', 'ignore') == b'-9.691e+05, -4.457e+06 (50.716617N, 12.267069W)'
    ax = plt.axes(projection=ccrs.PlateCarree())
    x, y = np.array([-181.5, 50.0])
    r = ax.format_coord(x, y)
    assert r.encode('ascii', 'ignore') == b'-181.5, 50 (50.000000N, 178.500000E)'
    ax = plt.axes(projection=ccrs.Robinson())
    x, y = np.array([16060595.2, 2363093.4])
    r = ax.format_coord(x, y)
    assert re.search(b'1.606e\\+07, 2.363e\\+06 \\(22.09[0-9]{4}N, 173.70[0-9]{4}E\\)', r.encode('ascii', 'ignore'))