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
def test_series_color_no_index(self):
    colors_ord = pd.Series(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'])
    ax1 = self.df.plot(colors_ord)
    self.df['colors_ord'] = colors_ord
    ax2 = self.df.plot('colors_ord')
    point_colors1 = ax1.collections[0].get_facecolors()
    point_colors2 = ax2.collections[0].get_facecolors()
    np.testing.assert_array_equal(point_colors1[1], point_colors2[1])