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
def test_subplots_norm(self):
    cmap = matplotlib.cm.viridis_r
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    ax = self.df.plot(column='values', cmap=cmap, norm=norm)
    actual_colors_orig = ax.collections[0].get_facecolors()
    exp_colors = cmap(np.arange(2) / 10)
    np.testing.assert_array_equal(exp_colors, actual_colors_orig)
    fig, ax = plt.subplots()
    self.df[1:].plot(column='values', ax=ax, norm=norm, cmap=cmap)
    actual_colors_sub = ax.collections[0].get_facecolors()
    np.testing.assert_array_equal(actual_colors_orig[1], actual_colors_sub[0])