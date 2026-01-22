import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import requires_scipy
def test_contour_plot_bounds():
    x = np.linspace(-2763217.0, 2681906.0, 200)
    y = np.linspace(-263790.62, 3230840.5, 130)
    data = np.hypot(*np.meshgrid(x, y)) / 200000.0
    proj_lcc = ccrs.LambertConformal(central_longitude=-95, central_latitude=25, standard_parallels=[25])
    ax = plt.axes(projection=proj_lcc)
    ax.contourf(x, y, data, levels=np.arange(0, 40, 1))
    assert_array_almost_equal(ax.get_extent(), np.array([x[0], x[-1], y[0], y[-1]]))
    plt.figure()
    ax = plt.axes(projection=proj_lcc)
    ax.contourf(x, y, data, levels=np.max(data) + np.arange(1, 3))