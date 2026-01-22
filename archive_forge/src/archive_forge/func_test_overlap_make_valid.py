import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
@pytest.mark.parametrize('make_valid', [True, False])
def test_overlap_make_valid(make_valid):
    bowtie = Polygon([(1, 1), (9, 9), (9, 1), (1, 9), (1, 1)])
    assert not bowtie.is_valid
    fixed_bowtie = bowtie.buffer(0)
    assert fixed_bowtie.is_valid
    df1 = GeoDataFrame({'col1': ['region'], 'geometry': GeoSeries([box(0, 0, 10, 10)])})
    df_bowtie = GeoDataFrame({'col1': ['invalid', 'valid'], 'geometry': GeoSeries([bowtie, fixed_bowtie])})
    if make_valid:
        df_overlay_bowtie = overlay(df1, df_bowtie, make_valid=make_valid)
        assert df_overlay_bowtie.at[0, 'geometry'].equals(fixed_bowtie)
        assert df_overlay_bowtie.at[1, 'geometry'].equals(fixed_bowtie)
    else:
        with pytest.raises(ValueError, match='1 invalid input geometries'):
            overlay(df1, df_bowtie, make_valid=make_valid)