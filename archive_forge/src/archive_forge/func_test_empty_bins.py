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
def test_empty_bins(self):
    bins = np.arange(1, 11) / 10
    ax = self.df.plot('low_vals', scheme='UserDefined', classification_kwds={'bins': bins}, legend=True)
    expected = np.array([[0.281412, 0.155834, 0.469201, 1.0], [0.267004, 0.004874, 0.329415, 1.0], [0.244972, 0.287675, 0.53726, 1.0]])
    assert all(((z == expected).all(axis=1).any() for z in ax.collections[0].get_facecolors()))
    labels = ['0.00, 0.10', '0.10, 0.20', '0.20, 0.30', '0.30, 0.40', '0.40, 0.50', '0.50, 0.60', '0.60, 0.70', '0.70, 0.80', '0.80, 0.90', '0.90, 1.00']
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == legend
    legend_colors_exp = [(0.267004, 0.004874, 0.329415, 1.0), (0.281412, 0.155834, 0.469201, 1.0), (0.244972, 0.287675, 0.53726, 1.0), (0.190631, 0.407061, 0.556089, 1.0), (0.147607, 0.511733, 0.557049, 1.0), (0.119699, 0.61849, 0.536347, 1.0), (0.20803, 0.718701, 0.472873, 1.0), (0.430983, 0.808473, 0.346476, 1.0), (0.709898, 0.868751, 0.169257, 1.0), (0.993248, 0.906157, 0.143936, 1.0)]
    assert [line.get_markerfacecolor() for line in ax.get_legend().get_lines()] == legend_colors_exp
    ax2 = self.df.plot('mid_vals', scheme='UserDefined', classification_kwds={'bins': bins}, legend=True)
    expected = np.array([[0.244972, 0.287675, 0.53726, 1.0], [0.190631, 0.407061, 0.556089, 1.0], [0.147607, 0.511733, 0.557049, 1.0], [0.119699, 0.61849, 0.536347, 1.0], [0.20803, 0.718701, 0.472873, 1.0]])
    assert all(((z == expected).all(axis=1).any() for z in ax2.collections[0].get_facecolors()))
    labels = ['-inf, 0.10', '0.10, 0.20', '0.20, 0.30', '0.30, 0.40', '0.40, 0.50', '0.50, 0.60', '0.60, 0.70', '0.70, 0.80', '0.80, 0.90', '0.90, 1.00']
    legend = [t.get_text() for t in ax2.get_legend().get_texts()]
    assert labels == legend
    assert [line.get_markerfacecolor() for line in ax2.get_legend().get_lines()] == legend_colors_exp
    ax3 = self.df.plot('high_vals', scheme='UserDefined', classification_kwds={'bins': bins}, legend=True)
    expected = np.array([[0.709898, 0.868751, 0.169257, 1.0], [0.993248, 0.906157, 0.143936, 1.0], [0.430983, 0.808473, 0.346476, 1.0]])
    assert all(((z == expected).all(axis=1).any() for z in ax3.collections[0].get_facecolors()))
    legend = [t.get_text() for t in ax3.get_legend().get_texts()]
    assert labels == legend
    assert [line.get_markerfacecolor() for line in ax3.get_legend().get_lines()] == legend_colors_exp