import toolz
import toolz.curried
from toolz.curried import (take, first, second, sorted, merge_with, reduce,
from collections import defaultdict
from importlib import import_module
from operator import add
def test_curried_operator():
    import operator
    for k, v in vars(cop).items():
        if not callable(v):
            continue
        if not isinstance(v, toolz.curry):
            try:
                v(1)
            except TypeError:
                try:
                    v('x')
                except TypeError:
                    pass
                else:
                    continue
                raise AssertionError('toolz.curried.operator.%s is not curried!' % k)
        assert should_curry(getattr(operator, k)) == isinstance(v, toolz.curry), k
    assert len(set(vars(cop)) & {'add', 'sub', 'mul'}) == 3