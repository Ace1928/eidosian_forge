import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_custom_colormaps(self):
    step = StepColormap(['green', 'yellow', 'red'], vmin=0, vmax=100000000)
    m = self.world.explore('pop_est', cmap=step, tooltip=['name'], legend=True)
    strings = ['fillColor":"#008000ff"', '"fillColor":"#ffff00ff"', '"fillColor":"#ff0000ff"']
    out_str = self._fetch_map_string(m)
    for s in strings:
        assert s in out_str
    assert out_str.count('008000ff') == 304
    assert out_str.count('ffff00ff') == 188
    assert out_str.count('ff0000ff') == 191

    def my_color_function(field):
        """Maps low values to green and high values to red."""
        if field > 100000000:
            return '#ff0000'
        else:
            return '#008000'
    m = self.world.explore('pop_est', cmap=my_color_function, legend=False)
    strings = ['"color":"#ff0000","fillColor":"#ff0000"', '"color":"#008000","fillColor":"#008000"']
    for s in strings:
        assert s in self._fetch_map_string(m)
    cmap = colors.ListedColormap(['red', 'green', 'blue', 'white', 'black'])
    m = self.nybb.explore('BoroName', cmap=cmap)
    strings = ['"fillColor":"#ff0000"', '"fillColor":"#008000"', '"fillColor":"#0000ff"', '"fillColor":"#ffffff"', '"fillColor":"#000000"']
    out_str = self._fetch_map_string(m)
    for s in strings:
        assert s in out_str