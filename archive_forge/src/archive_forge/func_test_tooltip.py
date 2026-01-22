import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_tooltip(self):
    """Test tooltip"""
    m = self.world.explore()
    assert 'GeoJsonTooltip' in str(m.to_dict())
    assert 'GeoJsonPopup' not in str(m.to_dict())
    m = self.world.explore(tooltip=True, popup=True)
    assert 'GeoJsonTooltip' in str(m.to_dict())
    assert 'GeoJsonPopup' in str(m.to_dict())
    out_str = self._fetch_map_string(m)
    assert 'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out_str
    assert 'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out_str
    m = self.world.explore(column='pop_est', tooltip=True, popup=True)
    assert 'GeoJsonTooltip' in str(m.to_dict())
    assert 'GeoJsonPopup' in str(m.to_dict())
    out_str = self._fetch_map_string(m)
    assert 'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out_str
    assert 'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out_str
    m = self.world.explore(tooltip='pop_est', popup='iso_a3')
    out_str = self._fetch_map_string(m)
    assert 'fields=["pop_est"]' in out_str
    assert 'aliases=["pop_est"]' in out_str
    assert 'fields=["iso_a3"]' in out_str
    assert 'aliases=["iso_a3"]' in out_str
    m = self.world.explore(tooltip=['pop_est', 'continent'], popup=['iso_a3', 'gdp_md_est'])
    out_str = self._fetch_map_string(m)
    assert 'fields=["pop_est","continent"]' in out_str
    assert 'aliases=["pop_est","continent"]' in out_str
    assert 'fields=["iso_a3","gdp_md_est"' in out_str
    assert 'aliases=["iso_a3","gdp_md_est"]' in out_str
    m = self.world.explore(tooltip=2, popup=2)
    out_str = self._fetch_map_string(m)
    assert 'fields=["pop_est","continent"]' in out_str
    assert 'aliases=["pop_est","continent"]' in out_str
    m = self.world.explore(tooltip=True, popup=False, tooltip_kwds={'aliases': [0, 1, 2, 3, 4, 5], 'sticky': False})
    out_str = self._fetch_map_string(m)
    assert 'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out_str
    assert 'aliases=[0,1,2,3,4,5]' in out_str
    assert '"sticky":false' in out_str
    m = self.world.explore(tooltip=False, popup=True, popup_kwds={'aliases': [0, 1, 2, 3, 4, 5]})
    out_str = self._fetch_map_string(m)
    assert 'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out_str
    assert 'aliases=[0,1,2,3,4,5]' in out_str
    assert '<th>${aliases[i]' in out_str
    m = self.world.explore(tooltip=True, popup=True, tooltip_kwds={'labels': False}, popup_kwds={'labels': False})
    out_str = self._fetch_map_string(m)
    assert '<th>${aliases[i]' not in out_str
    gdf = self.nybb.set_index('BoroName')
    m = gdf.explore()
    out_str = self._fetch_map_string(m)
    assert 'BoroName' in out_str