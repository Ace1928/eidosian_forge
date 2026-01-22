import warnings
from sympy.testing.pytest import (raises, warns, ignore_warnings,
from sympy.utilities.exceptions import sympy_deprecation_warning
def test_ignore_does_not_raise_without_warning():
    with warnings.catch_warnings(record=True) as w:
        with ignore_warnings(UserWarning):
            pass
        assert len(w) == 0