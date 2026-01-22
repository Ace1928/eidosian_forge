from unittest import SkipTest
import numpy as np
from matplotlib.colors import ListedColormap
from holoviews.element import Image, ImageStack, Raster
from holoviews.plotting.mpl.raster import RGBPlot
from .test_plot import TestMPLPlot, mpl_renderer
def test_raster_nodata_uint(self):
    arr = np.array([[0, 1, 2], [3, 4, 5]], dtype='uint32')
    expected = np.array([[3, 4, 5], [np.nan, 1, 2]])
    raster = Raster(arr).opts(nodata=0)
    plot = mpl_renderer.get_plot(raster)
    artist = plot.handles['artist']
    np.testing.assert_equal(artist.get_array().data, expected)