import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
def test_change_default_value():

    @change_default_value('arg1', new_value=-1, changed_version='0.12')
    def foo(arg0, arg1=0, arg2=1):
        """Expected docstring"""
        return (arg0, arg1, arg2)

    @change_default_value('arg1', new_value=-1, changed_version='0.12', warning_msg='Custom warning message')
    def bar(arg0, arg1=0, arg2=1):
        """Expected docstring"""
        return (arg0, arg1, arg2)
    with pytest.warns(FutureWarning) as record:
        assert foo(0) == (0, 0, 1)
        assert bar(0) == (0, 0, 1)
    expected_msg = 'The new recommended value for arg1 is -1. Until version 0.12, the default arg1 value is 0. From version 0.12, the arg1 default value will be -1. To avoid this warning, please explicitly set arg1 value.'
    assert str(record[0].message) == expected_msg
    assert str(record[1].message) == 'Custom warning message'
    with warnings.catch_warnings(record=True) as recorded:
        assert foo(0, 2) == (0, 2, 1)
        assert foo(0, arg1=0) == (0, 0, 1)
        assert foo.__name__ == 'foo'
        if sys.flags.optimize < 2:
            assert foo.__doc__ == 'Expected docstring'
    assert len(recorded) == 0