import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_read_postgis_chunksize(self, connection_postgis, df_nybb):
    """Test chunksize argument"""
    chunksize = 2
    con = connection_postgis
    create_postgis(con, df_nybb)
    sql = 'SELECT * FROM nybb;'
    df = pd.concat(read_postgis(sql, con, chunksize=chunksize))
    validate_boro_df(df)
    assert df.crs is None