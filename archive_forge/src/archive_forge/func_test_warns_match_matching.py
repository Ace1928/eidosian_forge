import warnings
from sympy.testing.pytest import (raises, warns, ignore_warnings,
from sympy.utilities.exceptions import sympy_deprecation_warning
def test_warns_match_matching():
    with warnings.catch_warnings(record=True) as w:
        with warns(UserWarning, match='this is the warning message'):
            warnings.warn('this is the warning message', UserWarning)
        assert len(w) == 0