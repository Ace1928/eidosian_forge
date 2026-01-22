import warnings
from sympy.testing.pytest import (raises, warns, ignore_warnings,
from sympy.utilities.exceptions import sympy_deprecation_warning
def test_ignore_continues_after_warning():
    with warnings.catch_warnings(record=True) as w:
        finished = False
        with ignore_warnings(UserWarning):
            warnings.warn('this is the warning message')
            finished = True
        assert finished
        assert len(w) == 0