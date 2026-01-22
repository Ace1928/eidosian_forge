import os
import zipfile
from dataclasses import dataclass
from typing import Any, Generator, List, NamedTuple, Optional, Tuple, Union
from urllib import request
import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import Generator as RNG
from scipy import sparse
import xgboost
from xgboost.data import pandas_pyarrow_mapper
def pd_arrow_dtypes() -> Generator:
    """Pandas DataFrame with pyarrow backed type."""
    import pandas as pd
    import pyarrow as pa
    dtypes = pandas_pyarrow_mapper
    Null: Union[float, None, Any] = np.nan
    orig = pd.DataFrame({'f0': [1, 2, Null, 3], 'f1': [4, 3, Null, 1]}, dtype=np.float32)
    for Null in (None, pd.NA):
        for dtype in dtypes:
            if dtype.startswith('float16') or dtype.startswith('bool'):
                continue
            df = pd.DataFrame({'f0': [1, 2, Null, 3], 'f1': [4, 3, Null, 1]}, dtype=dtype)
            yield (orig, df)
    orig = pd.DataFrame({'f0': [True, False, pd.NA, True], 'f1': [False, True, pd.NA, True]}, dtype=pd.BooleanDtype())
    df = pd.DataFrame({'f0': [True, False, pd.NA, True], 'f1': [False, True, pd.NA, True]}, dtype=pd.ArrowDtype(pa.bool_()))
    yield (orig, df)