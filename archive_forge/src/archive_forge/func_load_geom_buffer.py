import warnings
from contextlib import contextmanager
import pandas as pd
import shapely
import shapely.wkb
from geopandas import GeoDataFrame
from geopandas import _compat as compat
def load_geom_buffer(x):
    """Load from Python 2 binary."""
    return shapely.wkb.loads(str(x))