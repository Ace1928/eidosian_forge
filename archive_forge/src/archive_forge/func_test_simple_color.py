import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_simple_color(self):
    """Check color settings"""
    m = self.nybb.explore(color='red')
    out_str = self._fetch_map_string(m)
    assert '"fillColor":"red"' in out_str
    colors = ['#333333', '#367324', '#95824f', '#fcaa00', '#ffcc33']
    m2 = self.nybb.explore(color=colors)
    out_str = self._fetch_map_string(m2)
    for c in colors:
        assert f'"fillColor":"{c}"' in out_str
    df = self.nybb.copy()
    df['colors'] = colors
    m3 = df.explore(color='colors')
    out_str = self._fetch_map_string(m3)
    for c in colors:
        assert f'"fillColor":"{c}"' in out_str
    m4 = self.nybb.boundary.explore(color='red')
    out_str = self._fetch_map_string(m4)
    assert '"fillColor":"red"' in out_str