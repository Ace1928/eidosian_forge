import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_escape_special_characters(self):
    gdf = self.world.copy()
    gdf['name'] = '{{{what a mess}}} they are so different.'
    m = gdf.explore()
    out_str = self._fetch_map_string(m)
    assert '{{{' in out_str
    assert '}}}' in out_str