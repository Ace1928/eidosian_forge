import warnings
from sympy.testing.pytest import (raises, warns, ignore_warnings,
from sympy.utilities.exceptions import sympy_deprecation_warning
def test_expected_exception_is_silent_with():
    with raises(ValueError):
        raise ValueError()