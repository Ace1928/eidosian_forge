import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_categorical_legend(self):
    m = self.world.explore('continent', legend=True)
    out_str = self._fetch_map_string(m)
    assert "#1f77b4'></span>Africa" in out_str
    assert "#ff7f0e'></span>Antarctica" in out_str
    assert "#98df8a'></span>Asia" in out_str
    assert "#9467bd'></span>Europe" in out_str
    assert "#c49c94'></span>NorthAmerica" in out_str
    assert "#7f7f7f'></span>Oceania" in out_str
    assert "#dbdb8d'></span>Sevenseas(openocean)" in out_str
    assert "#9edae5'></span>SouthAmerica" in out_str
    m = self.missing.explore('continent', legend=True, missing_kwds={'color': 'red'})
    out_str = self._fetch_map_string(m)
    assert "red'></span>NaN" in out_str