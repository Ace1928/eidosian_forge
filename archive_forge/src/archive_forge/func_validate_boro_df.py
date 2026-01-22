import os.path
from pandas import Series
from geopandas import GeoDataFrame
from geopandas.testing import (  # noqa: F401
def validate_boro_df(df, case_sensitive=False):
    """Tests a GeoDataFrame that has been read in from the nybb dataset."""
    assert isinstance(df, GeoDataFrame)
    assert len(df) == 5
    columns = ('BoroCode', 'BoroName', 'Shape_Leng', 'Shape_Area')
    if case_sensitive:
        for col in columns:
            assert col in df.columns
    else:
        for col in columns:
            assert col.lower() in (dfcol.lower() for dfcol in df.columns)
    assert Series(df.geometry.geom_type).dropna().eq('MultiPolygon').all()