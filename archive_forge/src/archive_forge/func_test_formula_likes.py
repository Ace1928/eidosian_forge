import sys
import __future__
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc, Term, INTERCEPT
from patsy.categorical import C
from patsy.contrasts import Helmert
from patsy.user_util import balanced, LookupFactor
from patsy.build import (design_matrix_builders,
from patsy.highlevel import *
from patsy.util import (have_pandas,
from patsy.origin import Origin
def test_formula_likes():
    t([[1, 2, 3], [4, 5, 6]], {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['x0', 'x1', 'x2'])
    t((None, [[1, 2, 3], [4, 5, 6]]), {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['x0', 'x1', 'x2'])
    t(np.asarray([[1, 2, 3], [4, 5, 6]]), {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['x0', 'x1', 'x2'])
    t((None, np.asarray([[1, 2, 3], [4, 5, 6]])), {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['x0', 'x1', 'x2'])
    dm = DesignMatrix([[1, 2, 3], [4, 5, 6]], default_column_prefix='foo')
    t(dm, {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['foo0', 'foo1', 'foo2'])
    t((None, dm), {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['foo0', 'foo1', 'foo2'])
    t(([1, 2], [[1, 2, 3], [4, 5, 6]]), {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['x0', 'x1', 'x2'], [[1], [2]], ['y0'])
    t(([[1], [2]], [[1, 2, 3], [4, 5, 6]]), {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['x0', 'x1', 'x2'], [[1], [2]], ['y0'])
    t((np.asarray([1, 2]), np.asarray([[1, 2, 3], [4, 5, 6]])), {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['x0', 'x1', 'x2'], [[1], [2]], ['y0'])
    t((np.asarray([[1], [2]]), np.asarray([[1, 2, 3], [4, 5, 6]])), {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['x0', 'x1', 'x2'], [[1], [2]], ['y0'])
    x_dm = DesignMatrix([[1, 2, 3], [4, 5, 6]], default_column_prefix='foo')
    y_dm = DesignMatrix([1, 2], default_column_prefix='bar')
    t((y_dm, x_dm), {}, 0, False, [[1, 2, 3], [4, 5, 6]], ['foo0', 'foo1', 'foo2'], [[1], [2]], ['bar0'])
    t_invalid(([1, 2, 3], [[1, 2, 3], [4, 5, 6]]), {}, 0)
    t_invalid(([[1, 2, 3]],), {}, 0)
    t_invalid(([[1, 2, 3]], [[1, 2, 3]], [[1, 2, 3]]), {}, 0)
    if have_pandas:
        t(pandas.DataFrame({'x': [1, 2, 3]}), {}, 0, False, [[1], [2], [3]], ['x'])
        t(pandas.Series([1, 2, 3], name='asdf'), {}, 0, False, [[1], [2], [3]], ['asdf'])
        t((pandas.DataFrame({'y': [4, 5, 6]}), pandas.DataFrame({'x': [1, 2, 3]})), {}, 0, False, [[1], [2], [3]], ['x'], [[4], [5], [6]], ['y'])
        t((pandas.Series([4, 5, 6], name='y'), pandas.Series([1, 2, 3], name='x')), {}, 0, False, [[1], [2], [3]], ['x'], [[4], [5], [6]], ['y'])
        t((pandas.DataFrame([[4, 5, 6]]), pandas.DataFrame([[1, 2, 3]], columns=[7, 8, 9])), {}, 0, False, [[1, 2, 3]], ['x7', 'x8', 'x9'], [[4, 5, 6]], ['y0', 'y1', 'y2'])
        t(pandas.Series([1, 2, 3]), {}, 0, False, [[1], [2], [3]], ['x0'])
        t_invalid((pandas.DataFrame([[1]], index=[1]), pandas.DataFrame([[1]], index=[2])), {}, 0)

    class ForeignModelSource(object):

        def __patsy_get_model_desc__(self, data):
            return ModelDesc([Term([LookupFactor('Y')])], [Term([LookupFactor('X')])])
    foreign_model = ForeignModelSource()
    t(foreign_model, {'Y': [1, 2], 'X': [[1, 2], [3, 4]]}, 0, True, [[1, 2], [3, 4]], ['X[0]', 'X[1]'], [[1], [2]], ['Y'])

    class BadForeignModelSource(object):

        def __patsy_get_model_desc__(self, data):
            return data
    t_invalid(BadForeignModelSource(), {}, 0)
    t('y ~ x', {'y': [1, 2], 'x': [3, 4]}, 0, True, [[1, 3], [1, 4]], ['Intercept', 'x'], [[1], [2]], ['y'])
    t('~ x', {'y': [1, 2], 'x': [3, 4]}, 0, True, [[1, 3], [1, 4]], ['Intercept', 'x'])
    t('x + y', {'y': [1, 2], 'x': [3, 4]}, 0, True, [[1, 3, 1], [1, 4, 2]], ['Intercept', 'x', 'y'])
    if not six.PY3:
        t(unicode('y ~ x'), {'y': [1, 2], 'x': [3, 4]}, 0, True, [[1, 3], [1, 4]], ['Intercept', 'x'], [[1], [2]], ['y'])
        eacute = 'Ã©'.decode('utf-8')
        assert isinstance(eacute, unicode)
        pytest.raises(PatsyError, dmatrix, eacute, data={eacute: [1, 2]})
    desc = ModelDesc([], [Term([LookupFactor('x')])])
    t(desc, {'x': [1.5, 2.5, 3.5]}, 0, True, [[1.5], [2.5], [3.5]], ['x'])
    desc = ModelDesc([], [Term([]), Term([LookupFactor('x')])])
    t(desc, {'x': [1.5, 2.5, 3.5]}, 0, True, [[1, 1.5], [1, 2.5], [1, 3.5]], ['Intercept', 'x'])
    desc = ModelDesc([Term([LookupFactor('y')])], [Term([]), Term([LookupFactor('x')])])
    t(desc, {'x': [1.5, 2.5, 3.5], 'y': [10, 20, 30]}, 0, True, [[1, 1.5], [1, 2.5], [1, 3.5]], ['Intercept', 'x'], [[10], [20], [30]], ['y'])
    termlists = ([], [Term([LookupFactor('x')])], [Term([]), Term([LookupFactor('x')])])
    builders = design_matrix_builders(termlists, lambda: iter([{'x': [1, 2, 3]}]), eval_env=0)
    t((builders[0], builders[2]), {'x': [10, 20, 30]}, 0, True, [[1, 10], [1, 20], [1, 30]], ['Intercept', 'x'])
    t(builders[2], {'x': [10, 20, 30]}, 0, True, [[1, 10], [1, 20], [1, 30]], ['Intercept', 'x'])
    t((builders[1], builders[2]), {'x': [10, 20, 30]}, 0, True, [[1, 10], [1, 20], [1, 30]], ['Intercept', 'x'], [[10], [20], [30]], ['x'])
    x_in_env = [1, 2, 3]
    t('~ x_in_env', {}, 0, True, [[1, 1], [1, 2], [1, 3]], ['Intercept', 'x_in_env'])
    t('~ x_in_env', {'x_in_env': [10, 20, 30]}, 0, True, [[1, 10], [1, 20], [1, 30]], ['Intercept', 'x_in_env'])
    t_invalid('~ x_in_env', {}, 1, exc=(NameError, PatsyError))

    def check_nested_call():
        x_in_env = 'asdf'
        t('~ x_in_env', {}, 1, True, [[1, 1], [1, 2], [1, 3]], ['Intercept', 'x_in_env'])
    check_nested_call()
    e = EvalEnvironment.capture(1)
    t_invalid('~ x_in_env', {}, e, exc=(NameError, PatsyError))
    e = EvalEnvironment.capture(0)

    def check_nested_call_2():
        x_in_env = 'asdf'
        t('~ x_in_env', {}, e, True, [[1, 1], [1, 2], [1, 3]], ['Intercept', 'x_in_env'])
    check_nested_call_2()