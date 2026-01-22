import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_is_valid_curry():

    def check_curry(func, args, kwargs, incomplete=True):
        try:
            curry(func)(*args, **kwargs)
            curry(func, *args)(**kwargs)
            curry(func, **kwargs)(*args)
            curry(func, *args, **kwargs)()
            if not isinstance(func, type(lambda: None)):
                return None
            return incomplete
        except ValueError:
            return True
        except TypeError:
            return False
    check_valid = functools.partial(check_curry, incomplete=True)
    test_is_valid(check_valid=check_valid, incomplete=True)
    test_is_valid_py3(check_valid=check_valid, incomplete=True)
    check_valid = functools.partial(check_curry, incomplete=False)
    test_is_valid(check_valid=check_valid, incomplete=False)
    test_is_valid_py3(check_valid=check_valid, incomplete=False)