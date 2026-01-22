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
def get_ipy():
    try:
        ip = get_ipython()
    except:
        ip = None
    return ip