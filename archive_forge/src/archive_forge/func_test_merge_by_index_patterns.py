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
def test_merge_by_index_patterns(how, shuffle_method):
    pdf1l = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7], 'b': [7, 6, 5, 4, 3, 2, 1]})
    pdf1r = pd.DataFrame({'c': [1, 2, 3, 4, 5, 6, 7], 'd': [7, 6, 5, 4, 3, 2, 1]})
    pdf2l = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7], 'b': [7, 6, 5, 4, 3, 2, 1]}, index=list('abcdefg'))
    pdf2r = pd.DataFrame({'c': [7, 6, 5, 4, 3, 2, 1], 'd': [7, 6, 5, 4, 3, 2, 1]}, index=list('abcdefg'))
    pdf3l = pdf2l
    pdf3r = pd.DataFrame({'c': [6, 7, 8, 9], 'd': [5, 4, 3, 2]}, index=list('abdg'))
    pdf4l = pdf2l
    pdf4r = pd.DataFrame({'c': [9, 10, 11, 12], 'd': [5, 4, 3, 2]}, index=list('abdg'))
    pdf5l = pd.DataFrame({'a': [1, 1, 2, 2, 3, 3, 4], 'b': [7, 6, 5, 4, 3, 2, 1]}, index=list('lmnopqr'))
    pdf5r = pd.DataFrame({'c': [1, 1, 1, 1], 'd': [5, 4, 3, 2]}, index=list('abcd'))
    pdf6l = pd.DataFrame({'a': [1, 1, 2, 2, 3, 3, 4], 'b': [7, 6, 5, 4, 3, 2, 1]}, index=list('cdefghi'))
    pdf6r = pd.DataFrame({'c': [1, 2, 1, 2], 'd': [5, 4, 3, 2]}, index=list('abcd'))
    pdf7l = pd.DataFrame({'a': [1, 1, 2, 2, 3, 3, 4], 'b': [7, 6, 5, 4, 3, 2, 1]}, index=list('abcdefg'))
    pdf7r = pd.DataFrame({'c': [5, 6, 7, 8], 'd': [5, 4, 3, 2]}, index=list('fghi'))

    def fix_index(out, dtype):
        if len(out) == 0:
            return out.set_index(out.index.astype(dtype))
        return out
    for pdl, pdr in [(pdf1l, pdf1r), (pdf2l, pdf2r), (pdf3l, pdf3r), (pdf4l, pdf4r), (pdf5l, pdf5r), (pdf6l, pdf6r), (pdf7l, pdf7r)]:
        for lpart, rpart in [(2, 2), (3, 2), (2, 3)]:
            ddl = dd.from_pandas(pdl, lpart)
            ddr = dd.from_pandas(pdr, rpart)
            assert_eq(dd.merge(ddl, ddr, how=how, left_index=True, right_index=True, shuffle_method=shuffle_method), fix_index(pd.merge(pdl, pdr, how=how, left_index=True, right_index=True), pdl.index.dtype))
            assert_eq(dd.merge(ddr, ddl, how=how, left_index=True, right_index=True, shuffle_method=shuffle_method), fix_index(pd.merge(pdr, pdl, how=how, left_index=True, right_index=True), pdr.index.dtype))
            assert_eq(dd.merge(ddl, ddr, how=how, left_index=True, right_index=True, shuffle_method=shuffle_method, indicator=True), fix_index(pd.merge(pdl, pdr, how=how, left_index=True, right_index=True, indicator=True), pdl.index.dtype))
            assert_eq(dd.merge(ddr, ddl, how=how, left_index=True, right_index=True, shuffle_method=shuffle_method, indicator=True), fix_index(pd.merge(pdr, pdl, how=how, left_index=True, right_index=True, indicator=True), pdr.index.dtype))
            assert_eq(ddr.merge(ddl, how=how, left_index=True, right_index=True, shuffle_method=shuffle_method), fix_index(pdr.merge(pdl, how=how, left_index=True, right_index=True), pdr.index.dtype))
            assert_eq(ddl.merge(ddr, how=how, left_index=True, right_index=True, shuffle_method=shuffle_method), fix_index(pdl.merge(pdr, how=how, left_index=True, right_index=True), pdl.index.dtype))
            list_eq(dd.merge(ddl, ddr, how=how, left_on='a', right_on='c', shuffle_method=shuffle_method), pd.merge(pdl, pdr, how=how, left_on='a', right_on='c'))
            list_eq(dd.merge(ddl, ddr, how=how, left_on='b', right_on='d', shuffle_method=shuffle_method), pd.merge(pdl, pdr, how=how, left_on='b', right_on='d'))
            list_eq(dd.merge(ddr, ddl, how=how, left_on='c', right_on='a', shuffle_method=shuffle_method, indicator=True), pd.merge(pdr, pdl, how=how, left_on='c', right_on='a', indicator=True))
            list_eq(dd.merge(ddr, ddl, how=how, left_on='d', right_on='b', shuffle_method=shuffle_method, indicator=True), pd.merge(pdr, pdl, how=how, left_on='d', right_on='b', indicator=True))
            list_eq(dd.merge(ddr, ddl, how=how, left_on='c', right_on='a', shuffle_method=shuffle_method), pd.merge(pdr, pdl, how=how, left_on='c', right_on='a'))
            list_eq(dd.merge(ddr, ddl, how=how, left_on='d', right_on='b', shuffle_method=shuffle_method), pd.merge(pdr, pdl, how=how, left_on='d', right_on='b'))
            list_eq(ddl.merge(ddr, how=how, left_on='a', right_on='c', shuffle_method=shuffle_method), pdl.merge(pdr, how=how, left_on='a', right_on='c'))
            list_eq(ddl.merge(ddr, how=how, left_on='b', right_on='d', shuffle_method=shuffle_method), pdl.merge(pdr, how=how, left_on='b', right_on='d'))
            list_eq(ddr.merge(ddl, how=how, left_on='c', right_on='a', shuffle_method=shuffle_method), pdr.merge(pdl, how=how, left_on='c', right_on='a'))
            list_eq(ddr.merge(ddl, how=how, left_on='d', right_on='b', shuffle_method=shuffle_method), pdr.merge(pdl, how=how, left_on='d', right_on='b'))