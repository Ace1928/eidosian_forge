import warnings
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from geopandas import _vectorized
def assert_geodataframe_equal(left, right, check_dtype=True, check_index_type='equiv', check_column_type='equiv', check_frame_type=True, check_like=False, check_less_precise=False, check_geom_type=False, check_crs=True, normalize=False):
    """
    Check that two GeoDataFrames are equal/

    Parameters
    ----------
    left, right : two GeoDataFrames
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type, check_column_type : bool, default 'equiv'
        Check that index types are equal.
    check_frame_type : bool, default True
        Check that both are same type (*and* are GeoDataFrames). If False,
        will attempt to convert both into GeoDataFrame.
    check_like : bool, default False
        If true, ignore the order of rows & columns
    check_less_precise : bool, default False
        If True, use geom_equals_exact. if False, use geom_equals.
    check_geom_type : bool, default False
        If True, check that all the geom types are equal.
    check_crs: bool, default True
        If `check_frame_type` is True, then also check that the
        crs matches.
    normalize: bool, default False
        If True, normalize the geometries before comparing equality.
        Typically useful with ``check_less_precise=True``, which uses
        ``geom_equals_exact`` and requires exact coordinate order.
    """
    try:
        from pandas.testing import assert_frame_equal, assert_index_equal
    except ImportError:
        from pandas.util.testing import assert_frame_equal, assert_index_equal
    if check_frame_type:
        assert isinstance(left, GeoDataFrame)
        assert isinstance(left, type(right))
        if check_crs:
            if left._geometry_column_name is None and right._geometry_column_name is None:
                pass
            elif left._geometry_column_name not in left.columns and right._geometry_column_name not in right.columns:
                pass
            elif not left.crs and (not right.crs):
                pass
            else:
                assert left.crs == right.crs
    else:
        if not isinstance(left, GeoDataFrame):
            left = GeoDataFrame(left)
        if not isinstance(right, GeoDataFrame):
            right = GeoDataFrame(right)
    assert left.shape == right.shape, 'GeoDataFrame shape mismatch, left: {lshape!r}, right: {rshape!r}.\nLeft columns: {lcols!r}, right columns: {rcols!r}'.format(lshape=left.shape, rshape=right.shape, lcols=left.columns, rcols=right.columns)
    if check_like:
        left, right = (left.reindex_like(right), right)
    assert_index_equal(left.columns, right.columns, exact=check_column_type, obj='GeoDataFrame.columns')
    for col, dtype in left.dtypes.items():
        if isinstance(dtype, GeometryDtype):
            assert_geoseries_equal(left[col], right[col], normalize=normalize, check_dtype=check_dtype, check_less_precise=check_less_precise, check_geom_type=check_geom_type, check_crs=check_crs)
    assert left._geometry_column_name == right._geometry_column_name
    left2 = left.select_dtypes(exclude='geometry')
    right2 = right.select_dtypes(exclude='geometry')
    assert_frame_equal(left2, right2, check_dtype=check_dtype, check_index_type=check_index_type, check_column_type=check_column_type, obj='GeoDataFrame')