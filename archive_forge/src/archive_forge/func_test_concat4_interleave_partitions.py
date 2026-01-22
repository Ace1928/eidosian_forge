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
def test_concat4_interleave_partitions():
    pdf1 = pd.DataFrame(np.random.randn(10, 5), columns=list('ABCDE'), index=list('abcdefghij'))
    pdf2 = pd.DataFrame(np.random.randn(13, 5), columns=list('ABCDE'), index=list('fghijklmnopqr'))
    pdf3 = pd.DataFrame(np.random.randn(13, 6), columns=list('CDEXYZ'), index=list('fghijklmnopqr'))
    ddf1 = dd.from_pandas(pdf1, 2)
    ddf2 = dd.from_pandas(pdf2, 3)
    ddf3 = dd.from_pandas(pdf3, 2)
    cases = [[ddf1, ddf1], [ddf1, ddf2], [ddf1, ddf3], [ddf2, ddf1], [ddf2, ddf3], [ddf3, ddf1], [ddf3, ddf2]]
    for case in cases:
        pdcase = [c.compute() for c in case]
        assert_eq(dd.concat(case, interleave_partitions=True), pd.concat(pdcase, sort=False))
        assert_eq(dd.concat(case, join='inner', interleave_partitions=True), pd.concat(pdcase, join='inner', sort=False))
    msg = "'join' must be 'inner' or 'outer'"
    with pytest.raises(ValueError) as err:
        dd.concat([ddf1, ddf1], join='invalid', interleave_partitions=True)
    assert msg in str(err.value)