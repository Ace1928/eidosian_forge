from __future__ import annotations
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_220, PANDAS_GE_300
from dask.dataframe._pyarrow import is_object_string_dtype
from dask.dataframe.core import tokenize
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.utils import random_state_data
def make_categorical(n, rstate, choices=None, nunique=None, **kwargs):
    kwargs.pop('args', None)
    if nunique is not None:
        cat_len = len(str(nunique))
        choices = [str(x + 1).zfill(cat_len) for x in range(nunique)]
    else:
        choices = choices or names
    return pd.Categorical.from_codes(rstate.randint(0, len(choices), size=n), choices)