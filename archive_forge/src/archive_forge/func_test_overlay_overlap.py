import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_overlay_overlap(how):
    """
    Overlay test with overlapping geometries in both dataframes.
    Test files are created with::

        import geopandas
        from geopandas import GeoSeries, GeoDataFrame
        from shapely.geometry import Point, Polygon, LineString

        s1 = GeoSeries([Point(0, 0), Point(1.5, 0)]).buffer(1, resolution=2)
        s2 = GeoSeries([Point(1, 1), Point(2, 2)]).buffer(1, resolution=2)

        df1 = GeoDataFrame({'geometry': s1, 'col1':[1,2]})
        df2 = GeoDataFrame({'geometry': s2, 'col2':[1, 2]})

        ax = df1.plot(alpha=0.5)
        df2.plot(alpha=0.5, ax=ax, color='C1')

        df1.to_file('geopandas/geopandas/tests/data/df1_overlap.geojson',
                    driver='GeoJSON')
        df2.to_file('geopandas/geopandas/tests/data/df2_overlap.geojson',
                    driver='GeoJSON')

    and then overlay results are obtained from using  QGIS 2.16
    (Vector -> Geoprocessing Tools -> Intersection / Union / ...),
    saved to GeoJSON.
    """
    df1 = read_file(os.path.join(DATA, 'overlap', 'df1_overlap.geojson'))
    df2 = read_file(os.path.join(DATA, 'overlap', 'df2_overlap.geojson'))
    result = overlay(df1, df2, how=how)
    if how == 'identity':
        raise pytest.skip()
    expected = read_file(os.path.join(DATA, 'overlap', 'df1_df2_overlap-{0}.geojson'.format(how)))
    if how == 'union':
        expected = expected.iloc[:-1]
    result = result.reset_index(drop=True)
    if how == 'union':
        result = result.sort_values(['col1', 'col2']).reset_index(drop=True)
    assert_geodataframe_equal(result, expected, normalize=True, check_column_type=False, check_less_precise=True)