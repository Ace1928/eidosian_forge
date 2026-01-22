import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_is_valid_py3(check_valid=is_valid_args, incomplete=False):
    orig_check_valid = check_valid
    check_valid = lambda func, *args, **kwargs: orig_check_valid(func, args, kwargs)
    f = make_func('x, *, y=1')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1)
    assert check_valid(f, x=1)
    assert check_valid(f, 1, y=2)
    assert check_valid(f, 1, 2) is False
    assert check_valid(f, 1, z=2) is False
    f = make_func('x, *args, y=1')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1)
    assert check_valid(f, x=1)
    assert check_valid(f, 1, y=2)
    assert check_valid(f, 1, 2, y=2)
    assert check_valid(f, 1, 2)
    assert check_valid(f, 1, z=2) is False
    f = make_func('*, y=1')
    assert check_valid(f)
    assert check_valid(f, 1) is False
    assert check_valid(f, y=1)
    assert check_valid(f, z=1) is False
    f = make_func('x, *, y')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1) is incomplete
    assert check_valid(f, x=1) is incomplete
    assert check_valid(f, 1, y=2)
    assert check_valid(f, x=1, y=2)
    assert check_valid(f, 1, 2) is False
    assert check_valid(f, 1, z=2) is False
    assert check_valid(f, 1, y=1, z=2) is False
    f = make_func('x=1, *, y, z=3')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1, z=3) is incomplete
    assert check_valid(f, y=2)
    assert check_valid(f, 1, y=2)
    assert check_valid(f, x=1, y=2)
    assert check_valid(f, x=1, y=2, z=3)
    assert check_valid(f, 1, x=1, y=2) is False
    assert check_valid(f, 1, 3, y=2) is False
    f = make_func('w, x=2, *args, y, z=4')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1) is incomplete
    assert check_valid(f, 1, y=3)
    f = make_func('a, b, c=3, d=4, *args, e=5, f=6, g, h')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1) is incomplete
    assert check_valid(f, 1, 2) is incomplete
    assert check_valid(f, 1, 2, g=7) is incomplete
    assert check_valid(f, 1, 2, g=7, h=8)
    assert check_valid(f, 1, 2, 3, 4, 5, 6, 7, 8, 9) is incomplete
    f = make_func('a: int, b: float')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1) is incomplete
    assert check_valid(f, b=1) is incomplete
    assert check_valid(f, 1, 2)
    f = make_func('(a: int, b: float) -> float')
    assert check_valid(f) is incomplete
    assert check_valid(f, 1) is incomplete
    assert check_valid(f, b=1) is incomplete
    assert check_valid(f, 1, 2)
    f.__signature__ = 34
    assert check_valid(f) is False

    class RaisesValueError(object):

        def __call__(self):
            pass

        @property
        def __signature__(self):
            raise ValueError('Testing Python 3.4')
    f = RaisesValueError()
    assert check_valid(f) is None