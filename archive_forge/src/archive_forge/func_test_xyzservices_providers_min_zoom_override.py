import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_xyzservices_providers_min_zoom_override(self):
    xyzservices = pytest.importorskip('xyzservices')
    m = self.nybb.explore(tiles=xyzservices.providers.CartoDB.PositronNoLabels, min_zoom=3)
    out_str = self._fetch_map_string(m)
    assert '"maxNativeZoom":20,"maxZoom":20,"minZoom":3' in out_str