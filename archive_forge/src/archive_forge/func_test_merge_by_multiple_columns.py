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
@pytest.mark.slow
@pytest.mark.parametrize('how', ['inner', 'outer', 'left', 'right'])
def test_merge_by_multiple_columns(how, shuffle_method):

    def fix_index(out, dtype):
        if len(out) == 0:
            return out.set_index(out.index.astype(dtype))
        return out
    pdf1l = pd.DataFrame({'a': list('abcdefghij'), 'b': list('abcdefghij'), 'c': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=list('abcdefghij'))
    pdf1r = pd.DataFrame({'d': list('abcdefghij'), 'e': list('abcdefghij'), 'f': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}, index=list('abcdefghij'))
    pdf2l = pd.DataFrame({'a': list('abcdeabcde'), 'b': list('abcabcabca'), 'c': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=list('abcdefghij'))
    pdf2r = pd.DataFrame({'d': list('edcbaedcba'), 'e': list('aaabbbcccd'), 'f': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}, index=list('fghijklmno'))
    pdf3l = pd.DataFrame({'a': list('aaaaaaaaaa'), 'b': list('aaaaaaaaaa'), 'c': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=list('abcdefghij'))
    pdf3r = pd.DataFrame({'d': list('aaabbbccaa'), 'e': list('abbbbbbbbb'), 'f': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}, index=list('ABCDEFGHIJ'))
    for pdl, pdr in [(pdf1l, pdf1r), (pdf2l, pdf2r), (pdf3l, pdf3r)]:
        for lpart, rpart in [(2, 2), (3, 2), (2, 3)]:
            ddl = dd.from_pandas(pdl, lpart)
            ddr = dd.from_pandas(pdr, rpart)
            assert_eq(ddl.join(ddr, how=how, shuffle_method=shuffle_method), fix_index(pdl.join(pdr, how=how), pdl.index.dtype))
            assert_eq(ddr.join(ddl, how=how, shuffle_method=shuffle_method), fix_index(pdr.join(pdl, how=how), pdr.index.dtype))
            assert_eq(dd.merge(ddl, ddr, how=how, left_index=True, right_index=True, shuffle_method=shuffle_method), fix_index(pd.merge(pdl, pdr, how=how, left_index=True, right_index=True), pdl.index.dtype))
            assert_eq(dd.merge(ddr, ddl, how=how, left_index=True, right_index=True, shuffle_method=shuffle_method), fix_index(pd.merge(pdr, pdl, how=how, left_index=True, right_index=True), pdr.index.dtype))
            list_eq(dd.merge(ddl, ddr, how=how, left_on='a', right_on='d', shuffle_method=shuffle_method), pd.merge(pdl, pdr, how=how, left_on='a', right_on='d'))
            list_eq(dd.merge(ddl, ddr, how=how, left_on='b', right_on='e', shuffle_method=shuffle_method), pd.merge(pdl, pdr, how=how, left_on='b', right_on='e'))
            list_eq(dd.merge(ddr, ddl, how=how, left_on='d', right_on='a', shuffle_method=shuffle_method), pd.merge(pdr, pdl, how=how, left_on='d', right_on='a'))
            list_eq(dd.merge(ddr, ddl, how=how, left_on='e', right_on='b', shuffle_method=shuffle_method), pd.merge(pdr, pdl, how=how, left_on='e', right_on='b'))
            list_eq(dd.merge(ddl, ddr, how=how, left_on=['a', 'b'], right_on=['d', 'e'], shuffle_method=shuffle_method), pd.merge(pdl, pdr, how=how, left_on=['a', 'b'], right_on=['d', 'e']))