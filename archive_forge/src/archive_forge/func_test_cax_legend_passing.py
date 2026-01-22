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
def test_cax_legend_passing(self):
    """Pass a 'cax' argument to 'df.plot(.)', that is valid only if 'ax' is
        passed as well (if not, a new figure is created ad hoc, and 'cax' is
        ignored)
        """
    ax = plt.axes()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    with pytest.raises(ValueError):
        ax = self.df.plot(column='pop_est', cmap='OrRd', legend=True, cax=cax)