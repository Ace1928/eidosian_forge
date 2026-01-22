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
def test_style_kwargs_path_effects(self):
    from matplotlib.patheffects import withStroke
    effects = [withStroke(linewidth=8, foreground='b')]
    ax = self.df.plot(color='orange', path_effects=effects)
    assert ax.collections[0].get_path_effects()[0].__dict__['_gc'] == {'linewidth': 8, 'foreground': 'b'}