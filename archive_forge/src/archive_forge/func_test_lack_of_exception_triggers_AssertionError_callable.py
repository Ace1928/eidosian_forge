import warnings
from sympy.testing.pytest import (raises, warns, ignore_warnings,
from sympy.utilities.exceptions import sympy_deprecation_warning
def test_lack_of_exception_triggers_AssertionError_callable():
    try:
        raises(Exception, lambda: 1 + 1)
        assert False
    except Failed as e:
        assert 'DID NOT RAISE' in str(e)