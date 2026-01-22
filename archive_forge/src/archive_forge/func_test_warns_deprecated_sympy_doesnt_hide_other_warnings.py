import warnings
from sympy.testing.pytest import (raises, warns, ignore_warnings,
from sympy.utilities.exceptions import sympy_deprecation_warning
def test_warns_deprecated_sympy_doesnt_hide_other_warnings():
    with raises(RuntimeWarning):
        with warns_deprecated_sympy():
            _warn_sympy_deprecation()
            warnings.warn('this is the other message', RuntimeWarning)