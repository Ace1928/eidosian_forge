import itertools
from packaging.version import Version
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import (
from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.datasets import get_path
import geopandas._compat as compat
from geopandas.plotting import GeoplotAccessor
import pytest
import matplotlib.pyplot as plt
def test_default_colors(self):
    ax = self.points.plot()
    _check_colors(self.N, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * self.N)
    ax = self.df.plot()
    _check_colors(self.N, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * self.N)
    ax = self.df.plot(column='values')
    cmap = plt.get_cmap()
    expected_colors = cmap(np.arange(self.N) / (self.N - 1))
    _check_colors(self.N, ax.collections[0].get_facecolors(), expected_colors)