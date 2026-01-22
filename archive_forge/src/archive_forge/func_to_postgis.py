import json
import warnings
import numpy as np
import pandas as pd
import shapely.errors
from pandas import DataFrame, Series
from pandas.core.accessor import CachedAccessor
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
from geopandas.array import GeometryArray, GeometryDtype, from_shapely, to_wkb, to_wkt
from geopandas.base import GeoPandasBase, is_geometry_type
from geopandas.geoseries import GeoSeries
import geopandas.io
from geopandas.explore import _explore
from . import _compat as compat
from ._decorator import doc
def to_postgis(self, name, con, schema=None, if_exists='fail', index=False, index_label=None, chunksize=None, dtype=None):
    """
        Upload GeoDataFrame into PostGIS database.

        This method requires SQLAlchemy and GeoAlchemy2, and a PostgreSQL
        Python driver (e.g. psycopg2) to be installed.

        It is also possible to use :meth:`~GeoDataFrame.to_file` to write to a database.
        Especially for file geodatabases like GeoPackage or SpatiaLite this can be
        easier.

        Parameters
        ----------
        name : str
            Name of the target table.
        con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
            Active connection to the PostGIS database.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            How to behave if the table already exists:

            - fail: Raise a ValueError.
            - replace: Drop the table before inserting new values.
            - append: Insert new values to the existing table.
        schema : string, optional
            Specify the schema. If None, use default schema: 'public'.
        index : bool, default False
            Write DataFrame index as a column.
            Uses *index_label* as the column name in the table.
        index_label : string or sequence, default None
            Column label for index column(s).
            If None is given (default) and index is True,
            then the index names are used.
        chunksize : int, optional
            Rows will be written in batches of this size at a time.
            By default, all rows will be written at once.
        dtype : dict of column name to SQL type, default None
            Specifying the datatype for columns.
            The keys should be the column names and the values
            should be the SQLAlchemy types.

        Examples
        --------

        >>> from sqlalchemy import create_engine
        >>> engine = create_engine("postgresql://myusername:mypassword@myhost:5432/mydatabase")  # doctest: +SKIP
        >>> gdf.to_postgis("my_table", engine)  # doctest: +SKIP

        See also
        --------
        GeoDataFrame.to_file : write GeoDataFrame to file
        read_postgis : read PostGIS database to GeoDataFrame

        """
    geopandas.io.sql._write_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)