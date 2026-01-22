from __future__ import annotations
import contextlib
import operator
import warnings
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import _concat
from dask.dataframe.utils import (
def test_categorize():
    pdf = frames4[0]
    if pyarrow_strings_enabled():
        pdf = to_pyarrow_string(pdf)
    meta = clear_known_categories(pdf).rename(columns={'y': 'y_'})
    dsk = {('unknown', i): df for i, df in enumerate(frames3)}
    if not dd._dask_expr_enabled():
        ddf = dd.DataFrame(dsk, 'unknown', make_meta(meta, parent_meta=frames[0]), [None] * 4).repartition(npartitions=10).rename(columns={'y': 'y_'})
    else:
        pdf = pd.concat(dsk.values()).rename(columns={'y': 'y_'}).astype({'w': 'category', 'y_': 'category'})
        pdf.index = pdf.index.astype('category')
        ddf = dd.from_pandas(pdf, npartitions=4, sort=False)
        ddf['w'] = ddf.w.cat.as_unknown()
        ddf['y_'] = ddf.y_.cat.as_unknown()
        ddf.index = ddf.index.cat.as_unknown()
    ddf = ddf.assign(w=ddf.w.cat.set_categories(['x', 'y', 'z']))
    assert ddf.w.cat.known
    assert not ddf.y_.cat.known
    assert not ddf.index.cat.known
    df = ddf.compute()
    for index in [None, True, False]:
        known_index = index is not False
        ddf2 = ddf.categorize(index=index)
        assert ddf2.y_.cat.known
        assert ddf2.v.cat.known
        assert ddf2.index.cat.known == known_index
        assert_eq(ddf2, df.astype({'v': 'category'}), check_categorical=False)
        ddf2 = ddf.categorize(index=index, split_every=2)
        assert ddf2.y_.cat.known
        assert ddf2.v.cat.known
        assert ddf2.index.cat.known == known_index
        assert_eq(ddf2, df.astype({'v': 'category'}), check_categorical=False)
        ddf2 = ddf.categorize('v', index=index)
        assert not ddf2.y_.cat.known
        assert ddf2.v.cat.known
        assert ddf2.index.cat.known == known_index
        assert_eq(ddf2, df.astype({'v': 'category'}), check_categorical=False)
        ddf2 = ddf.categorize('y_', index=index)
        assert ddf2.y_.cat.known
        assert ddf2.v.dtype == get_string_dtype()
        assert ddf2.index.cat.known == known_index
        assert_eq(ddf2, df)
    ddf_known_index = ddf.categorize(columns=[], index=True)
    assert ddf_known_index.index.cat.known
    assert_eq(ddf_known_index, df)
    assert ddf.categorize(['w'], index=False) is ddf
    assert ddf.categorize([], index=False) is ddf
    assert ddf_known_index.categorize(['w']) is ddf_known_index
    assert ddf_known_index.categorize([]) is ddf_known_index
    with pytest.raises(ValueError):
        ddf.categorize(split_every=1)
    with pytest.raises(ValueError):
        ddf.categorize(split_every='foo')