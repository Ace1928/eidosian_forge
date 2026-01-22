import sys
from collections.abc import Hashable
from functools import wraps
from packaging.version import Version
from types import FunctionType
import bokeh
import numpy as np
import pandas as pd
import param
import holoviews as hv
def is_geodataframe(data):
    if 'spatialpandas' in sys.modules:
        import spatialpandas as spd
        if isinstance(data, spd.GeoDataFrame):
            return True
    return isinstance(data, pd.DataFrame) and hasattr(data, 'geom_type') and hasattr(data, 'geometry')