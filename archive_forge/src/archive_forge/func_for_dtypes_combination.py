import functools
import inspect
import os
import random
from typing import Tuple, Type
import traceback
import unittest
import warnings
import numpy
import cupy
from cupy.testing import _array
from cupy.testing import _parameterized
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def for_dtypes_combination(types, names=('dtype',), full=None):
    """Decorator that checks the fixture with a product set of dtypes.

    Args:
         types(list of dtypes): dtypes to be tested.
         names(list of str): Argument names to which dtypes are passed.
         full(bool): If ``True``, then all combinations
             of dtypes will be tested.
             Otherwise, the subset of combinations will be tested
             (see the description below).

    Decorator adds the keyword arguments specified by ``names``
    to the test fixture. Then, it runs the fixtures in parallel
    with passing (possibly a subset of) the product set of dtypes.
    The range of dtypes is specified by ``types``.

    The combination of dtypes to be tested changes depending
    on the option ``full``. If ``full`` is ``True``,
    all combinations of ``types`` are tested.
    Sometimes, such an exhaustive test can be costly.
    So, if ``full`` is ``False``, only a subset of possible combinations
    is randomly sampled. If ``full`` is ``None``, the behavior is
    determined by an environment variable ``CUPY_TEST_FULL_COMBINATION``.
    If the value is set to ``'1'``, it behaves as if ``full=True``, and
    otherwise ``full=False``.
    """
    types = list(types)
    if len(types) == 1:
        name, = names
        return for_dtypes(types, name)
    if full is None:
        full = int(os.environ.get('CUPY_TEST_FULL_COMBINATION', '0')) != 0
    if full:
        combination = _parameterized.product({name: types for name in names})
    else:
        ts = []
        for _ in range(len(names)):
            shuffled_types = types[:]
            random.shuffle(shuffled_types)
            ts.append(types + shuffled_types)
        combination = [tuple(zip(names, typs)) for typs in zip(*ts)]
        combination = [dict(assoc_list) for assoc_list in set(combination)]

    def decorator(impl):

        @_wraps_partial(impl, *names)
        def test_func(*args, **kw):
            for dtypes in combination:
                kw_copy = kw.copy()
                kw_copy.update(dtypes)
                try:
                    impl(*args, **kw_copy)
                except _skip_classes as e:
                    msg = ', '.join(('{} = {}'.format(name, dtype) for name, dtype in dtypes.items()))
                    print('skipped: {} ({})'.format(msg, e))
                except Exception:
                    print(dtypes)
                    raise
        return test_func
    return decorator