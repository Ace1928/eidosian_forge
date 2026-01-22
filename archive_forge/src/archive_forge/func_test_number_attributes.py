from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('code,expected', [['int.numerator', int.numerator], ['float.is_integer', float.is_integer], ['complex.real', complex.real]])
def test_number_attributes(code, expected):
    assert guarded_eval(code, limited()) == expected