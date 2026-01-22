import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
@pytest.mark.parametrize('geom_types', ['polys', 'poly_line', 'poly_point', 'line_poly', 'point_poly'])
def test_overlay_strict(how, keep_geom_type, geom_types):
    """
    Test of mixed geometry types on input and output. Expected results initially
    generated using following snippet.

        polys1 = gpd.GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
        df1 = gpd.GeoDataFrame({'col1': [1, 2], 'geometry': polys1})

        polys2 = gpd.GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
        df2 = gpd.GeoDataFrame({'geometry': polys2, 'col2': [1, 2, 3]})

        lines1 = gpd.GeoSeries([LineString([(2, 0), (2, 4), (6, 4)]),
                                LineString([(0, 3), (6, 3)])])
        df3 = gpd.GeoDataFrame({'col3': [1, 2], 'geometry': lines1})
        points1 = gpd.GeoSeries([Point((2, 2)),
                                 Point((3, 3))])
        df4 = gpd.GeoDataFrame({'col4': [1, 2], 'geometry': points1})

        params=["union", "intersection", "difference", "symmetric_difference",
                "identity"]
        stricts = [True, False]

        for p in params:
            for s in stricts:
                exp = gpd.overlay(df1, df2, how=p, keep_geom_type=s)
                if not exp.empty:
                    exp.to_file('polys_{p}_{s}.geojson'.format(p=p, s=s),
                                driver='GeoJSON')

        for p in params:
            for s in stricts:
                exp = gpd.overlay(df1, df3, how=p, keep_geom_type=s)
                if not exp.empty:
                    exp.to_file('poly_line_{p}_{s}.geojson'.format(p=p, s=s),
                                driver='GeoJSON')
        for p in params:
            for s in stricts:
                exp = gpd.overlay(df1, df4, how=p, keep_geom_type=s)
                if not exp.empty:
                    exp.to_file('poly_point_{p}_{s}.geojson'.format(p=p, s=s),
                                driver='GeoJSON')
    """
    polys1 = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
    df1 = GeoDataFrame({'col1': [1, 2], 'geometry': polys1})
    polys2 = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
    df2 = GeoDataFrame({'geometry': polys2, 'col2': [1, 2, 3]})
    lines1 = GeoSeries([LineString([(2, 0), (2, 4), (6, 4)]), LineString([(0, 3), (6, 3)])])
    df3 = GeoDataFrame({'col3': [1, 2], 'geometry': lines1})
    points1 = GeoSeries([Point((2, 2)), Point((3, 3))])
    df4 = GeoDataFrame({'col4': [1, 2], 'geometry': points1})
    if geom_types == 'polys':
        result = overlay(df1, df2, how=how, keep_geom_type=keep_geom_type)
    elif geom_types == 'poly_line':
        result = overlay(df1, df3, how=how, keep_geom_type=keep_geom_type)
    elif geom_types == 'poly_point':
        result = overlay(df1, df4, how=how, keep_geom_type=keep_geom_type)
    elif geom_types == 'line_poly':
        result = overlay(df3, df1, how=how, keep_geom_type=keep_geom_type)
    elif geom_types == 'point_poly':
        result = overlay(df4, df1, how=how, keep_geom_type=keep_geom_type)
    try:
        expected = read_file(os.path.join(DATA, 'strict', '{t}_{h}_{s}.geojson'.format(t=geom_types, h=how, s=keep_geom_type)))
        assert all(expected.columns == result.columns), 'Column name mismatch'
        cols = list(set(result.columns) - {'geometry'})
        expected = expected.sort_values(cols, axis=0).reset_index(drop=True)
        result = result.sort_values(cols, axis=0).reset_index(drop=True)
        assert_geodataframe_equal(result, expected, normalize=True, check_column_type=False, check_less_precise=True, check_crs=False, check_dtype=False)
    except DriverError:
        assert result.empty
    except OSError:
        assert result.empty
    except RuntimeError:
        assert result.empty