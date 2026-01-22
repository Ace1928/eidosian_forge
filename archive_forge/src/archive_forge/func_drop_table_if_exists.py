import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def drop_table_if_exists(conn_or_engine, table):
    sqlalchemy = pytest.importorskip('sqlalchemy')
    if sqlalchemy.inspect(conn_or_engine).has_table(table):
        metadata = sqlalchemy.MetaData()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="Did not recognize type 'geometry' of column.*")
            metadata.reflect(conn_or_engine)
        table = metadata.tables.get(table)
        if table is not None:
            table.drop(conn_or_engine, checkfirst=True)