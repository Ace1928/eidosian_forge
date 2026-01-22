from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize('func,sparse_output', [(do('all'), False), (do('any'), False), (do('astype', dtype=int), True), (do('clip', min=0, max=1), True), (do('coarsen', windows={'x': 2}, func='sum'), True), (do('compute'), True), (do('conj'), True), (do('copy'), True), (do('count'), False), (do('get_axis_num', dim='x'), False), (do('isel', x=slice(2, 4)), True), (do('isnull'), True), (do('load'), True), (do('mean'), False), (do('notnull'), True), (do('roll'), True), (do('round'), True), (do('set_dims', dims=('x', 'y', 'z')), True), (do('stack', dimensions={'flat': ('x', 'y')}), True), (do('to_base_variable'), True), (do('transpose'), True), (do('unstack', dimensions={'x': {'x1': 5, 'x2': 2}}), True), (do('broadcast_equals', make_xrvar({'x': 10, 'y': 5})), False), (do('equals', make_xrvar({'x': 10, 'y': 5})), False), (do('identical', make_xrvar({'x': 10, 'y': 5})), False), param(do('argmax'), True, marks=[xfail(reason='Missing implementation for np.argmin'), filterwarnings('ignore:Behaviour of argmin/argmax')]), param(do('argmin'), True, marks=[xfail(reason='Missing implementation for np.argmax'), filterwarnings('ignore:Behaviour of argmin/argmax')]), param(do('argsort'), True, marks=xfail(reason="'COO' object has no attribute 'argsort'")), param(do('concat', variables=[make_xrvar({'x': 10, 'y': 5}), make_xrvar({'x': 10, 'y': 5})]), True), param(do('conjugate'), True, marks=xfail(reason="'COO' object has no attribute 'conjugate'")), param(do('cumprod'), True, marks=xfail(reason='Missing implementation for np.nancumprod')), param(do('cumsum'), True, marks=xfail(reason='Missing implementation for np.nancumsum')), (do('fillna', 0), True), param(do('item', (1, 1)), False, marks=xfail(reason="'COO' object has no attribute 'item'")), param(do('median'), False, marks=xfail(reason='Missing implementation for np.nanmedian')), param(do('max'), False), param(do('min'), False), param(do('no_conflicts', other=make_xrvar({'x': 10, 'y': 5})), True, marks=xfail(reason='mixed sparse-dense operation')), param(do('pad', mode='constant', pad_widths={'x': (1, 1)}, fill_value=5), True, marks=xfail(reason='Missing implementation for np.pad')), (do('prod'), False), param(do('quantile', q=0.5), True, marks=xfail(reason='Missing implementation for np.nanpercentile')), param(do('rank', dim='x'), False, marks=xfail(reason='Only implemented for NumPy arrays (via bottleneck)')), param(do('reduce', func='sum', dim='x'), True), param(do('rolling_window', dim='x', window=2, window_dim='x_win'), True, marks=xfail(reason='Missing implementation for np.pad')), param(do('shift', x=2), True, marks=xfail(reason='mixed sparse-dense operation')), param(do('std'), False, marks=xfail(reason='Missing implementation for np.nanstd')), (do('sum'), False), param(do('var'), False, marks=xfail(reason='Missing implementation for np.nanvar')), param(do('to_dict'), False), (do('where', cond=make_xrvar({'x': 10, 'y': 5}) > 0.5), True)], ids=repr)
def test_variable_method(func, sparse_output):
    var_s = make_xrvar({'x': 10, 'y': 5})
    var_d = xr.Variable(var_s.dims, var_s.data.todense())
    ret_s = func(var_s)
    ret_d = func(var_d)
    if isinstance(ret_d, xr.Variable) and isinstance(ret_d.data, sparse.SparseArray):
        ret_d = ret_d.copy(data=ret_d.data.todense())
    if sparse_output:
        assert isinstance(ret_s.data, sparse.SparseArray)
        assert np.allclose(ret_s.data.todense(), ret_d.data, equal_nan=True)
    elif func.meth != 'to_dict':
        assert np.allclose(ret_s, ret_d)
    else:
        arr_s, arr_d = (ret_s.pop('data'), ret_d.pop('data'))
        assert np.allclose(arr_s, arr_d)
        assert ret_s == ret_d