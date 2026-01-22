from __future__ import print_function
import warnings
import numbers
import six
import numpy as np
from patsy import PatsyError
from patsy.util import atleast_2d_column_default
from patsy.compat import OrderedDict
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
from patsy.constraint import linear_constraint
from patsy.contrasts import ContrastMatrix
from patsy.desc import ModelDesc, Term
def test_DesignInfo():
    import pytest

    class _MockFactor(object):

        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name
    f_x = _MockFactor('x')
    f_y = _MockFactor('y')
    t_x = Term([f_x])
    t_y = Term([f_y])
    factor_infos = {f_x: FactorInfo(f_x, 'numerical', {}, num_columns=3), f_y: FactorInfo(f_y, 'numerical', {}, num_columns=1)}
    term_codings = OrderedDict([(t_x, [SubtermInfo([f_x], {}, 3)]), (t_y, [SubtermInfo([f_y], {}, 1)])])
    di = DesignInfo(['x1', 'x2', 'x3', 'y'], factor_infos, term_codings)
    assert di.column_names == ['x1', 'x2', 'x3', 'y']
    assert di.term_names == ['x', 'y']
    assert di.terms == [t_x, t_y]
    assert di.column_name_indexes == {'x1': 0, 'x2': 1, 'x3': 2, 'y': 3}
    assert di.term_name_slices == {'x': slice(0, 3), 'y': slice(3, 4)}
    assert di.term_slices == {t_x: slice(0, 3), t_y: slice(3, 4)}
    assert di.describe() == 'x + y'
    assert di.slice(1) == slice(1, 2)
    assert di.slice('x1') == slice(0, 1)
    assert di.slice('x2') == slice(1, 2)
    assert di.slice('x3') == slice(2, 3)
    assert di.slice('x') == slice(0, 3)
    assert di.slice(t_x) == slice(0, 3)
    assert di.slice('y') == slice(3, 4)
    assert di.slice(t_y) == slice(3, 4)
    assert di.slice(slice(2, 4)) == slice(2, 4)
    pytest.raises(PatsyError, di.slice, 'asdf')
    repr(di)
    assert_no_pickling(di)
    di = DesignInfo(['a1', 'a2', 'a3', 'b'])
    assert di.column_names == ['a1', 'a2', 'a3', 'b']
    assert di.term_names == ['a1', 'a2', 'a3', 'b']
    assert di.terms is None
    assert di.column_name_indexes == {'a1': 0, 'a2': 1, 'a3': 2, 'b': 3}
    assert di.term_name_slices == {'a1': slice(0, 1), 'a2': slice(1, 2), 'a3': slice(2, 3), 'b': slice(3, 4)}
    assert di.term_slices is None
    assert di.describe() == 'a1 + a2 + a3 + b'
    assert di.slice(1) == slice(1, 2)
    assert di.slice('a1') == slice(0, 1)
    assert di.slice('a2') == slice(1, 2)
    assert di.slice('a3') == slice(2, 3)
    assert di.slice('b') == slice(3, 4)
    assert DesignInfo(['Intercept', 'a', 'b']).describe() == '1 + a + b'
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y'], factor_infos=factor_infos)
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y'], term_codings=term_codings)
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y'], list(factor_infos), term_codings)
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y1', 'y2'], factor_infos, term_codings)
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3'], factor_infos, term_codings)
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'y', 'y2'], factor_infos, term_codings)
    pytest.raises(ValueError, DesignInfo, ['x1', 'x1', 'x1', 'y'], factor_infos, term_codings)
    term_codings_x_only = OrderedDict(term_codings)
    del term_codings_x_only[t_y]
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3'], factor_infos, term_codings_x_only)
    f_a = _MockFactor('a')
    t_a = Term([f_a])
    term_codings_with_a = OrderedDict(term_codings)
    term_codings_with_a[t_a] = [SubtermInfo([f_a], {}, 1)]
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y', 'a'], factor_infos, term_codings_with_a)
    not_factor_infos = dict(factor_infos)
    not_factor_infos[f_x] = "what is this I don't even"
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y'], not_factor_infos, term_codings)
    mismatch_factor_infos = dict(factor_infos)
    mismatch_factor_infos[f_x] = FactorInfo(f_a, 'numerical', {}, num_columns=3)
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y'], mismatch_factor_infos, term_codings)
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y'], factor_infos, dict(term_codings))
    not_term_codings = OrderedDict(term_codings)
    not_term_codings['this is a string'] = term_codings[t_x]
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y'], factor_infos, not_term_codings)
    non_list_term_codings = OrderedDict(term_codings)
    non_list_term_codings[t_y] = tuple(term_codings[t_y])
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y'], factor_infos, non_list_term_codings)
    non_subterm_term_codings = OrderedDict(term_codings)
    non_subterm_term_codings[t_y][0] = 'not a SubtermInfo'
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y'], factor_infos, non_subterm_term_codings)
    bad_subterm = OrderedDict(term_codings)
    term_codings[t_y][0] = SubtermInfo([f_x], {}, 1)
    pytest.raises(ValueError, DesignInfo, ['x1', 'x2', 'x3', 'y'], factor_infos, bad_subterm)
    factor_codings_a = {f_a: FactorInfo(f_a, 'categorical', {}, categories=['a1', 'a2'])}
    term_codings_a_bad_rows = OrderedDict([(t_a, [SubtermInfo([f_a], {f_a: ContrastMatrix(np.ones((3, 2)), ['[1]', '[2]'])}, 2)])])
    pytest.raises(ValueError, DesignInfo, ['a[1]', 'a[2]'], factor_codings_a, term_codings_a_bad_rows)
    t_ax = Term([f_a, f_x])
    factor_codings_ax = {f_a: FactorInfo(f_a, 'categorical', {}, categories=['a1', 'a2']), f_x: FactorInfo(f_x, 'numerical', {}, num_columns=2)}
    term_codings_ax_extra_cm = OrderedDict([(t_ax, [SubtermInfo([f_a, f_x], {f_a: ContrastMatrix(np.ones((2, 2)), ['[1]', '[2]']), f_x: ContrastMatrix(np.ones((2, 2)), ['[1]', '[2]'])}, 4)])])
    pytest.raises(ValueError, DesignInfo, ['a[1]:x[1]', 'a[2]:x[1]', 'a[1]:x[2]', 'a[2]:x[2]'], factor_codings_ax, term_codings_ax_extra_cm)
    term_codings_ax_missing_cm = OrderedDict([(t_ax, [SubtermInfo([f_a, f_x], {}, 4)])])
    pytest.raises((ValueError, KeyError), DesignInfo, ['a[1]:x[1]', 'a[2]:x[1]', 'a[1]:x[2]', 'a[2]:x[2]'], factor_codings_ax, term_codings_ax_missing_cm)
    term_codings_ax_wrong_subterm_columns = OrderedDict([(t_ax, [SubtermInfo([f_a, f_x], {f_a: ContrastMatrix(np.ones((2, 3)), ['[1]', '[2]', '[3]'])}, 5)])])
    pytest.raises(ValueError, DesignInfo, ['a[1]:x[1]', 'a[2]:x[1]', 'a[3]:x[1]', 'a[1]:x[2]', 'a[2]:x[2]', 'a[3]:x[2]'], factor_codings_ax, term_codings_ax_wrong_subterm_columns)