import warnings
from sympy.testing.pytest import (raises, warns, ignore_warnings,
from sympy.utilities.exceptions import sympy_deprecation_warning
def test_unexpected_exception_is_passed_through_with():
    try:
        with raises(TypeError):
            raise ValueError('some error message')
        assert False
    except ValueError as e:
        assert str(e) == 'some error message'