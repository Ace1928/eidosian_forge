import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_write_postgis_in_chunks(self, engine_postgis, df_mixed_single_and_multi):
    """
        Tests writing a LinearRing works.
        """
    engine = engine_postgis
    table = 'geomtype_tests'
    write_postgis(df_mixed_single_and_multi, con=engine, name=table, if_exists='replace', chunksize=1)
    sql = text('SELECT COUNT(geometry) FROM {table};'.format(table=table))
    with engine.connect() as conn:
        row_cnt = conn.execute(sql).fetchone()[0]
    assert row_cnt == 3
    sql = text('SELECT DISTINCT GeometryType(geometry) FROM {table} ORDER BY 1;'.format(table=table))
    with engine.connect() as conn:
        res = conn.execute(sql).fetchall()
    assert res[0][0].upper() == 'LINESTRING'
    assert res[1][0].upper() == 'MULTILINESTRING'
    assert res[2][0].upper() == 'POINT'