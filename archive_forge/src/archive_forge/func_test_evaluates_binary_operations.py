from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('code,expected', [['1 + 1', 2], ['3 - 1', 2], ['2 * 3', 6], ['5 // 2', 2], ['5 / 2', 2.5], ['5**2', 25], ['2 >> 1', 1], ['2 << 1', 4], ['1 | 2', 3], ['1 & 1', 1], ['1 & 2', 0]])
@pytest.mark.parametrize('context', LIMITED_OR_HIGHER)
def test_evaluates_binary_operations(code, expected, context):
    assert guarded_eval(code, context()) == expected