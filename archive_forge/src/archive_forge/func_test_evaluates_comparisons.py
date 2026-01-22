from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('code,expected', [['2 > 1', True], ['2 < 1', False], ['2 <= 1', False], ['2 <= 2', True], ['1 >= 2', False], ['2 >= 2', True], ['2 == 2', True], ['1 == 2', False], ['1 != 2', True], ['1 != 1', False], ['1 < 4 < 3', False], ['(1 < 4) < 3', True], ['4 > 3 > 2 > 1', True], ['4 > 3 > 2 > 9', False], ['1 < 2 < 3 < 4', True], ['9 < 2 < 3 < 4', False], ['1 < 2 > 1 > 0 > -1 < 1', True], ['1 in [1] in [[1]]', True], ['1 in [1] in [[2]]', False], ['1 in [1]', True], ['0 in [1]', False], ['1 not in [1]', False], ['0 not in [1]', True], ['True is True', True], ['False is False', True], ['True is False', False], ['True is not True', False], ['False is not True', True]])
@pytest.mark.parametrize('context', LIMITED_OR_HIGHER)
def test_evaluates_comparisons(code, expected, context):
    assert guarded_eval(code, context()) == expected