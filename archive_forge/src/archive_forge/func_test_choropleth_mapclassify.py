import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_choropleth_mapclassify(self):
    """Mapclassify bins"""
    m = self.nybb.explore(column='Shape_Leng', scheme='quantiles')
    out_str = self._fetch_map_string(m)
    assert 'color":"#21918c"' in out_str
    assert 'color":"#3b528b"' in out_str
    assert 'color":"#5ec962"' in out_str
    assert 'color":"#fde725"' in out_str
    assert 'color":"#440154"' in out_str
    m = self.world.explore(column='pop_est', scheme='headtailbreaks')
    out_str = self._fetch_map_string(m)
    assert '"fillColor":"#3b528b"' in out_str
    assert '"fillColor":"#21918c"' in out_str
    assert '"fillColor":"#5ec962"' in out_str
    assert '"fillColor":"#fde725"' in out_str
    assert '"fillColor":"#440154"' in out_str
    m = self.world.explore(column='pop_est', scheme='naturalbreaks', k=3)
    out_str = self._fetch_map_string(m)
    assert '"fillColor":"#21918c"' in out_str
    assert '"fillColor":"#fde725"' in out_str
    assert '"fillColor":"#440154"' in out_str
    m = self.chicago.explore(column='POP2010', scheme='UserDefined', classification_kwds={'bins': [25000, 50000, 75000, 100000]})
    out_str = self._fetch_map_string(m)
    assert '"fillColor":"#fde725"' in out_str
    assert '"fillColor":"#35b779"' in out_str
    assert '"fillColor":"#31688e"' in out_str
    assert '"fillColor":"#440154"' in out_str