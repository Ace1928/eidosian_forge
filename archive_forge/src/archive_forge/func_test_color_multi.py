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
def test_color_multi(self):
    ax = self.mgdf.plot(color=self.mgdf['color_rgba'])
    _check_colors(4, np.concatenate([c.get_edgecolor() for c in ax.collections]), ['green'] * 2 + ['blue'] * 2)
    _check_colors(4, np.concatenate([c.get_facecolor() for c in ax.collections]), ['red'] * 2 + ['blue'] * 2)