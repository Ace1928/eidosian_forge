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
def is_ibis(data):
    if not check_library(data, 'ibis'):
        return False
    import ibis
    return isinstance(data, ibis.Expr)