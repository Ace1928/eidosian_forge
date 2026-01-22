from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_evaluates_if_expression():
    context = limited()
    assert guarded_eval('2 if True else 3', context) == 2
    assert guarded_eval('4 if False else 5', context) == 5