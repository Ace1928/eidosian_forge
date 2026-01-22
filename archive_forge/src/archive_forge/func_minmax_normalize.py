from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
def minmax_normalize(X, lower, upper):
    return (X - lower) / (upper - lower)