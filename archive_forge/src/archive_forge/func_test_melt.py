from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.api.types import is_object_dtype
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.core import _Frame
from dask.dataframe.methods import concat
from dask.dataframe.multi import (
from dask.dataframe.utils import (
from dask.utils_test import hlg_layer, hlg_layer_topological
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason='melt not supported yet')
@pytest.mark.parametrize('kwargs', [{}, dict(id_vars='int'), dict(value_vars='int'), dict(value_vars=['obj', 'int'], var_name='myvar'), dict(id_vars='s1', value_vars=['obj', 'int'], value_name='myval'), dict(value_vars=['obj', 's1']), dict(value_vars=['s1', 's2'])])
def test_melt(kwargs):
    pdf = pd.DataFrame({'obj': list('abcd') * 5, 's1': list('XY') * 10, 's2': list('abcde') * 4, 'int': np.random.randn(20)})
    if pa:
        pdf = pdf.astype({'s1': 'string[pyarrow]', 's2': 'string[pyarrow]'})
    ddf = dd.from_pandas(pdf, 4)
    list_eq(dd.melt(ddf, **kwargs), pd.melt(pdf, **kwargs))
    list_eq(ddf.melt(**kwargs), pdf.melt(**kwargs))