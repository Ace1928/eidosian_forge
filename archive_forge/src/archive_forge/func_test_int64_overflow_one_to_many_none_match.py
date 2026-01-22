from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
@pytest.mark.slow
@pytest.mark.parametrize('how', ['left', 'right', 'outer', 'inner'])
@pytest.mark.parametrize('sort', [True, False])
def test_int64_overflow_one_to_many_none_match(self, how, sort):
    low, high, n = (-1 << 10, 1 << 10, 1 << 11)
    left = DataFrame(np.random.default_rng(2).integers(low, high, (n, 7)).astype('int64'), columns=list('ABCDEFG'))
    shape = left.apply(Series.nunique).values
    assert is_int64_overflow_possible(shape)
    left = concat([left, left], ignore_index=True)
    right = DataFrame(np.random.default_rng(3).integers(low, high, (n // 2, 7)).astype('int64'), columns=list('ABCDEFG'))
    i = np.random.default_rng(4).choice(len(left), n)
    right = concat([right, right, left.iloc[i]], ignore_index=True)
    left['left'] = np.random.default_rng(2).standard_normal(len(left))
    right['right'] = np.random.default_rng(2).standard_normal(len(right))
    i = np.random.default_rng(5).permutation(len(left))
    left = left.iloc[i].copy()
    left.index = np.arange(len(left))
    i = np.random.default_rng(6).permutation(len(right))
    right = right.iloc[i].copy()
    right.index = np.arange(len(right))
    ldict, rdict = (defaultdict(list), defaultdict(list))
    for idx, row in left.set_index(list('ABCDEFG')).iterrows():
        ldict[idx].append(row['left'])
    for idx, row in right.set_index(list('ABCDEFG')).iterrows():
        rdict[idx].append(row['right'])
    vals = []
    for k, lval in ldict.items():
        rval = rdict.get(k, [np.nan])
        for lv, rv in product(lval, rval):
            vals.append(k + (lv, rv))
    for k, rval in rdict.items():
        if k not in ldict:
            vals.extend((k + (np.nan, rv) for rv in rval))

    def align(df):
        df = df.sort_values(df.columns.tolist())
        df.index = np.arange(len(df))
        return df
    out = DataFrame(vals, columns=list('ABCDEFG') + ['left', 'right'])
    out = align(out)
    jmask = {'left': out['left'].notna(), 'right': out['right'].notna(), 'inner': out['left'].notna() & out['right'].notna(), 'outer': np.ones(len(out), dtype='bool')}
    mask = jmask[how]
    frame = align(out[mask].copy())
    assert mask.all() ^ mask.any() or how == 'outer'
    res = merge(left, right, how=how, sort=sort)
    if sort:
        kcols = list('ABCDEFG')
        tm.assert_frame_equal(res[kcols].copy(), res[kcols].sort_values(kcols, kind='mergesort'))
    tm.assert_frame_equal(frame, align(res))