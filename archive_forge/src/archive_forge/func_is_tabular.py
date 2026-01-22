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
def is_tabular(data):
    if check_library(data, ['dask', 'streamz', 'pandas', 'geopandas', 'cudf']):
        return True
    elif check_library(data, 'intake'):
        from intake.source.base import DataSource
        if isinstance(data, DataSource):
            return data.container == 'dataframe'
    else:
        return False