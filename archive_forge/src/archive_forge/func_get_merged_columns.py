from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
@staticmethod
def get_merged_columns(params):
    return ['src_x', 'src_y', 'dst_x', 'dst_y', params.weight]