import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_style_kwds(self):
    """Style keywords"""
    m = self.world.explore(style_kwds={'fillOpacity': 0.1, 'weight': 0.5, 'fillColor': 'orange'})
    out_str = self._fetch_map_string(m)
    assert '"fillColor":"orange","fillOpacity":0.1,"weight":0.5' in out_str
    m = self.world.explore(column='pop_est', style_kwds={'color': 'black'})
    assert '"color":"black"' in self._fetch_map_string(m)
    m = self.world.explore(style_kwds={'style_function': lambda x: {'fillColor': 'red' if x['properties']['gdp_md_est'] < 10 ** 6 else 'green', 'color': 'black' if x['properties']['gdp_md_est'] < 10 ** 6 else 'white'}})
    assert all(('"fillColor":"green"' in t and '"color":"white"' in t or ('"fillColor":"red"' in t and '"color":"black"' in t) for t in [''.join(line.split()) for line in m._parent.render().split('\n') if 'return' in line and 'color' in line]))
    with pytest.raises(ValueError, match="'style_function' has to be a callable"):
        self.world.explore(style_kwds={'style_function': 'not callable'})