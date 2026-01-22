import warnings
import pytest
from cirq.testing import assert_deprecated
def test_nested_assert_deprecation():
    with assert_deprecated(deadline='v1.2', count=1):
        with assert_deprecated(deadline='v1.2', count=1):
            with assert_deprecated(deadline='v1.2', count=1):
                warnings.warn('hello, this is deprecated in v1.2')