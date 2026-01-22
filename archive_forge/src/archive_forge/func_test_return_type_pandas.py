from __future__ import print_function
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.util import (atleast_2d_column_default,
from patsy.desc import Term, INTERCEPT
from patsy.build import *
from patsy.categorical import C
from patsy.user_util import balanced, LookupFactor
from patsy.design_info import DesignMatrix, DesignInfo
def test_return_type_pandas():
    if not have_pandas:
        return
    data = pandas.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'a': ['a1', 'a2', 'a1']}, index=[10, 20, 30])

    def iter_maker():
        yield data
    int_builder, = design_matrix_builders([make_termlist([])], iter_maker, 0)
    y_builder, x_builder = design_matrix_builders([make_termlist('y'), make_termlist('x')], iter_maker, eval_env=0)
    x_a_builder, = design_matrix_builders([make_termlist('x', 'a')], iter_maker, eval_env=0)
    x_y_builder, = design_matrix_builders([make_termlist('x', 'y')], iter_maker, eval_env=0)
    pytest.raises(PatsyError, build_design_matrices, [x_a_builder], {'x': data['x'], 'a': data['a'][::-1]})
    pytest.raises(PatsyError, build_design_matrices, [y_builder, x_builder], {'x': data['x'], 'y': data['y'][::-1]})

    class CheatingDataFrame(pandas.DataFrame):

        def __getitem__(self, key):
            if key == 'x':
                return pandas.DataFrame.__getitem__(self, key)[::-1]
            else:
                return pandas.DataFrame.__getitem__(self, key)
    pytest.raises(PatsyError, build_design_matrices, [x_builder], CheatingDataFrame(data))
    mat, = build_design_matrices([x_y_builder], {'x': data['x'], 'y': [40, 50, 60]})
    assert np.allclose(mat, [[1, 40], [2, 50], [3, 60]])
    y_df, x_df = build_design_matrices([y_builder, x_builder], data, return_type='dataframe')
    assert isinstance(y_df, pandas.DataFrame)
    assert isinstance(x_df, pandas.DataFrame)
    assert np.array_equal(y_df, [[4], [5], [6]])
    assert np.array_equal(x_df, [[1], [2], [3]])
    assert np.array_equal(y_df.index, [10, 20, 30])
    assert np.array_equal(x_df.index, [10, 20, 30])
    assert np.array_equal(y_df.columns, ['y'])
    assert np.array_equal(x_df.columns, ['x'])
    assert y_df.design_info.column_names == ['y']
    assert x_df.design_info.column_names == ['x']
    assert y_df.design_info.term_names == ['y']
    assert x_df.design_info.term_names == ['x']
    y_df, x_df = build_design_matrices([y_builder, x_builder], {'y': [7, 8, 9], 'x': data['x']}, return_type='dataframe')
    assert isinstance(y_df, pandas.DataFrame)
    assert isinstance(x_df, pandas.DataFrame)
    assert np.array_equal(y_df, [[7], [8], [9]])
    assert np.array_equal(x_df, [[1], [2], [3]])
    assert np.array_equal(y_df.index, [10, 20, 30])
    assert np.array_equal(x_df.index, [10, 20, 30])
    assert np.array_equal(y_df.columns, ['y'])
    assert np.array_equal(x_df.columns, ['x'])
    assert y_df.design_info.column_names == ['y']
    assert x_df.design_info.column_names == ['x']
    assert y_df.design_info.term_names == ['y']
    assert x_df.design_info.term_names == ['x']
    x_a_df, = build_design_matrices([x_a_builder], {'x': [-1, -2, -3], 'a': data['a']}, return_type='dataframe')
    assert isinstance(x_a_df, pandas.DataFrame)
    assert np.array_equal(x_a_df, [[1, 0, -1], [0, 1, -2], [1, 0, -3]])
    assert np.array_equal(x_a_df.index, [10, 20, 30])
    x_y_df, = build_design_matrices([x_y_builder], {'y': [7, 8, 9], 'x': [10, 11, 12]}, return_type='dataframe')
    assert isinstance(x_y_df, pandas.DataFrame)
    assert np.array_equal(x_y_df, [[10, 7], [11, 8], [12, 9]])
    assert np.array_equal(x_y_df.index, [0, 1, 2])
    int_df, = build_design_matrices([int_builder], data, return_type='dataframe')
    assert isinstance(int_df, pandas.DataFrame)
    assert np.array_equal(int_df, [[1], [1], [1]])
    assert int_df.index.equals(pandas.Index([10, 20, 30]))
    import patsy.build
    had_pandas = patsy.build.have_pandas
    try:
        patsy.build.have_pandas = False
        pytest.raises(PatsyError, build_design_matrices, [x_builder], {'x': [1, 2, 3]}, return_type='dataframe')
    finally:
        patsy.build.have_pandas = had_pandas
    x_df, = build_design_matrices([x_a_builder], {'x': [1.0, np.nan, 3.0], 'a': np.asarray([None, 'a2', 'a1'], dtype=object)}, NA_action='drop', return_type='dataframe')
    assert x_df.index.equals(pandas.Index([2]))