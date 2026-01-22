from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def rowmin(arr0, arr1):
    bigint = np.max([np.max(arr0), np.max(arr1)]) + 1
    arr0[arr0 < 0] = bigint
    arr1[arr1 < 0] = bigint
    ret = np.minimum(arr0, arr1)
    ret[ret == bigint] = -1
    return ret