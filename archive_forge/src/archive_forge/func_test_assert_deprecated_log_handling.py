import warnings
import pytest
from cirq.testing import assert_deprecated
def test_assert_deprecated_log_handling():
    with assert_deprecated('hello', deadline='v1.2'):
        warnings.warn('hello, this is deprecated in v1.2')
    with pytest.raises(AssertionError, match='Expected 1 log message but got 0.'):
        with assert_deprecated(deadline='v1.2'):
            pass
    with pytest.raises(AssertionError, match='Expected 1 log message but got 2.'):
        with assert_deprecated(deadline='v1.2'):
            warnings.warn('hello, this is deprecated in v1.2')
            warnings.warn('hello, this is deprecated in v1.2')
    with assert_deprecated(deadline='v1.2', count=2):
        warnings.warn('hello, this is deprecated in v1.2')
        warnings.warn('hello, this is deprecated in v1.2')
    with assert_deprecated(deadline='v1.2', count=None):
        warnings.warn('hello, this is deprecated in v1.2')
        warnings.warn('hello, this is deprecated in v1.2')
        warnings.warn('hello, this is deprecated in v1.2')