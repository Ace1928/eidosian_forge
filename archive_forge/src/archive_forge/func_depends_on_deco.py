import sys
import types
import inspect
from functools import wraps, update_wrapper
from sympy.utilities.exceptions import sympy_deprecation_warning
def depends_on_deco(fn):
    fn._doctest_depends_on = dependencies
    fn.__doctest_skip__ = skiptests
    if inspect.isclass(fn):
        fn._doctest_depdends_on = no_attrs_in_subclass(fn, fn._doctest_depends_on)
        fn.__doctest_skip__ = no_attrs_in_subclass(fn, fn.__doctest_skip__)
    return fn