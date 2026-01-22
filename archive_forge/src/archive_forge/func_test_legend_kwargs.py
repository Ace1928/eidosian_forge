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
def test_legend_kwargs(self):
    categories = list(self.df['values'].unique())
    prefix = 'LABEL_FOR_'
    ax = self.df.plot(column='values', categorical=True, categories=categories, legend=True, legend_kwds={'labels': [prefix + str(c) for c in categories], 'frameon': False})
    assert len(categories) == len(ax.get_legend().get_texts())
    assert ax.get_legend().get_texts()[0].get_text().startswith(prefix)
    assert ax.get_legend().get_frame_on() is False