import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
def test__examine_factor_types():
    from patsy.categorical import C

    class MockFactor(object):

        def __init__(self):
            from patsy.origin import Origin
            self.origin = Origin('MOCK', 1, 2)

        def eval(self, state, data):
            return state[data]

        def name(self):
            return 'MOCK MOCK'

    class DataIterMaker(object):

        def __init__(self):
            self.i = -1

        def __call__(self):
            return self

        def __iter__(self):
            return self

        def next(self):
            self.i += 1
            if self.i > 1:
                raise StopIteration
            return self.i
        __next__ = next
    num_1dim = MockFactor()
    num_1col = MockFactor()
    num_4col = MockFactor()
    categ_1col = MockFactor()
    bool_1col = MockFactor()
    string_1col = MockFactor()
    object_1col = MockFactor()
    object_levels = (object(), object(), object())
    factor_states = {num_1dim: ([1, 2, 3], [4, 5, 6]), num_1col: ([[1], [2], [3]], [[4], [5], [6]]), num_4col: (np.zeros((3, 4)), np.ones((3, 4))), categ_1col: (C(['a', 'b', 'c'], levels=('a', 'b', 'c'), contrast='MOCK CONTRAST'), C(['c', 'b', 'a'], levels=('a', 'b', 'c'), contrast='MOCK CONTRAST')), bool_1col: ([True, True, False], [False, True, True]), string_1col: (['a', 'a', 'a'], ['c', 'b', 'a']), object_1col: ([object_levels[0]] * 3, object_levels)}
    it = DataIterMaker()
    num_column_counts, cat_levels_contrasts = _examine_factor_types(factor_states.keys(), factor_states, it, NAAction())
    assert it.i == 2
    iterations = 0
    assert num_column_counts == {num_1dim: 1, num_1col: 1, num_4col: 4}
    assert cat_levels_contrasts == {categ_1col: (('a', 'b', 'c'), 'MOCK CONTRAST'), bool_1col: ((False, True), None), string_1col: (('a', 'b', 'c'), None), object_1col: (tuple(sorted(object_levels, key=id)), None)}
    it = DataIterMaker()
    no_read_necessary = [num_1dim, num_1col, num_4col, categ_1col, bool_1col]
    num_column_counts, cat_levels_contrasts = _examine_factor_types(no_read_necessary, factor_states, it, NAAction())
    assert it.i == 0
    assert num_column_counts == {num_1dim: 1, num_1col: 1, num_4col: 4}
    assert cat_levels_contrasts == {categ_1col: (('a', 'b', 'c'), 'MOCK CONTRAST'), bool_1col: ((False, True), None)}
    bool_3col = MockFactor()
    num_3dim = MockFactor()
    string_3col = MockFactor()
    object_3col = MockFactor()
    illegal_factor_states = {num_3dim: (np.zeros((3, 3, 3)), np.ones((3, 3, 3))), string_3col: ([['a', 'b', 'c']], [['b', 'c', 'a']]), object_3col: ([[[object()]]], [[[object()]]])}
    import pytest
    for illegal_factor in illegal_factor_states:
        it = DataIterMaker()
        try:
            _examine_factor_types([illegal_factor], illegal_factor_states, it, NAAction())
        except PatsyError as e:
            assert e.origin is illegal_factor.origin
        else:
            assert False