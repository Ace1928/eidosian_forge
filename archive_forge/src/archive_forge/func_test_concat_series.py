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
@pytest.mark.parametrize('join', ['inner', 'outer'])
def test_concat_series(join):
    pdf1 = pd.DataFrame({'x': [1, 2, 3, 4, 6, 7], 'y': list('abcdef')}, index=[1, 2, 3, 4, 6, 7])
    ddf1 = dd.from_pandas(pdf1, 2)
    pdf2 = pd.DataFrame({'x': [1, 2, 3, 4, 6, 7], 'y': list('abcdef')}, index=[8, 9, 10, 11, 12, 13])
    ddf2 = dd.from_pandas(pdf2, 2)
    pdf3 = pd.DataFrame({'x': [1, 2, 3, 4, 6, 7], 'z': list('abcdef')}, index=[8, 9, 10, 11, 12, 13])
    ddf3 = dd.from_pandas(pdf3, 2)
    kwargs = {'sort': False}
    for dd1, dd2, pd1, pd2 in [(ddf1.x, ddf2.x, pdf1.x, pdf2.x), (ddf1.x, ddf3.z, pdf1.x, pdf3.z)]:
        expected = pd.concat([pd1, pd2], join=join, **kwargs)
        result = dd.concat([dd1, dd2], join=join, **kwargs)
        assert_eq(result, expected)