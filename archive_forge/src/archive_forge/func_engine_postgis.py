import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
@pytest.fixture()
def engine_postgis():
    """
    Initiates a connection engine to a postGIS database that must already exist.
    """
    sqlalchemy = pytest.importorskip('sqlalchemy')
    from sqlalchemy.engine.url import URL
    user = os.environ.get('PGUSER')
    password = os.environ.get('PGPASSWORD')
    host = os.environ.get('PGHOST')
    port = os.environ.get('PGPORT')
    dbname = 'test_geopandas'
    try:
        con = sqlalchemy.create_engine(URL.create(drivername='postgresql+psycopg2', username=user, database=dbname, password=password, host=host, port=port))
        con.connect()
    except Exception:
        pytest.skip('Cannot connect with postgresql database')
    yield con
    con.dispose()