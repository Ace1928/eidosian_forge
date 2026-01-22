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
def is_cudf(data):
    if 'cudf' in sys.modules:
        from cudf import DataFrame, Series
        return isinstance(data, (DataFrame, Series))