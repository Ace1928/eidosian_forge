import warnings
from contextlib import contextmanager
import pandas as pd
import shapely
import shapely.wkb
from geopandas import GeoDataFrame
from geopandas import _compat as compat
def read_postgis(*args, **kwargs):
    import warnings
    warnings.warn('geopandas.io.sql.read_postgis() is intended for internal use only, and will be deprecated. Use geopandas.read_postgis() instead.', FutureWarning, stacklevel=2)
    return _read_postgis(*args, **kwargs)