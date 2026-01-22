from unittest import SkipTest
import numpy as np
from matplotlib.colors import ListedColormap
from holoviews.element import Image, ImageStack, Raster
from holoviews.plotting.mpl.raster import RGBPlot
from .test_plot import TestMPLPlot, mpl_renderer
def test_image_listed_cmap(self):
    colors = ['#ffffff', '#000000']
    img = Image(np.array([[0, 1, 2], [3, 4, 5]])).opts(cmap=colors)
    plot = mpl_renderer.get_plot(img)
    artist = plot.handles['artist']
    cmap = artist.get_cmap()
    self.assertIsInstance(cmap, ListedColormap)
    assert cmap.colors == colors