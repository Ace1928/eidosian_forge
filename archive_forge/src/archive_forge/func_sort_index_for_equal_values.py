import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def sort_index_for_equal_values(df, ascending=True):
    """Sort `df` indices of equal rows."""
    if df.index.dtype == np.float64:
        df.index = df.index.astype('str')
    res = df.groupby(by=df if df.ndim == 1 else df.columns, sort=False).apply(lambda df: df.sort_index(ascending=ascending))
    if res.index.nlevels > df.index.nlevels:
        res.index = res.index.droplevel(0)
    res.index.names = df.index.names
    return res