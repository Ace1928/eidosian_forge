import sys
import types
import inspect
from functools import wraps, update_wrapper
from sympy.utilities.exceptions import sympy_deprecation_warning
def skiptests():
    from sympy.testing.runtests import DependencyError, SymPyDocTests, PyTestReporter
    r = PyTestReporter()
    t = SymPyDocTests(r, None)
    try:
        t._check_dependencies(**dependencies)
    except DependencyError:
        return True
    else:
        return False