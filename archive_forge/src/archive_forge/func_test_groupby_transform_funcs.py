from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.parametrize('transformation', [lambda x: x.sum(), np.sum, 'sum', pd.Series.rank])
def test_groupby_transform_funcs(transformation):
    pdf = pd.DataFrame({'A': [1, 2, 3, 4] * 5, 'B': np.random.randn(20), 'C': np.random.randn(20), 'D': np.random.randn(20)})
    ddf = dd.from_pandas(pdf, 3)
    with pytest.warns(UserWarning):
        assert_eq(pdf.groupby('A').transform(transformation), ddf.groupby('A').transform(transformation))
        assert_eq(pdf.groupby('A')['B'].transform(transformation), ddf.groupby('A')['B'].transform(transformation))