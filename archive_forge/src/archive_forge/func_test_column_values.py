import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_column_values(self):
    """
        Check that the dataframe plot method returns same values with an
        input string (column in df), pd.Series, or np.array
        """
    column_array = np.array(self.world['pop_est'])
    m1 = self.world.explore(column='pop_est')
    m2 = self.world.explore(column=column_array)
    m3 = self.world.explore(column=self.world['pop_est'])
    assert m1.location == m2.location == m3.location
    m1_fields = self.world.explore(column=column_array, tooltip=True, popup=True)
    out1_fields_str = self._fetch_map_string(m1_fields)
    assert 'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out1_fields_str
    assert 'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out1_fields_str
    m2_fields = self.world.explore(column=self.world['pop_est'], tooltip=True, popup=True)
    out2_fields_str = self._fetch_map_string(m2_fields)
    assert 'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out2_fields_str
    assert 'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out2_fields_str
    with pytest.raises(ValueError, match='different number of rows'):
        self.world.explore(column=np.array([1, 2, 3]))