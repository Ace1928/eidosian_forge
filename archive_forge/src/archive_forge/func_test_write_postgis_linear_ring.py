import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_write_postgis_linear_ring(self, engine_postgis, df_linear_ring):
    """
        Tests that writing a LinearRing.
        """
    engine = engine_postgis
    table = 'geomtype_tests'
    write_postgis(df_linear_ring, con=engine, name=table, if_exists='replace')
    sql = text('SELECT DISTINCT(GeometryType(geometry)) FROM {table} ORDER BY 1;'.format(table=table))
    with engine.connect() as conn:
        geom_type = conn.execute(sql).fetchone()[0]
    assert geom_type.upper() == 'LINESTRING'