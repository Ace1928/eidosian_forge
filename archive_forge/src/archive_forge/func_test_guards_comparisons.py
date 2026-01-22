from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_guards_comparisons():

    class GoodEq(int):
        pass

    class BadEq(int):

        def __eq__(self, other):
            assert False
    context = limited(bad=BadEq(1), good=GoodEq(1))
    with pytest.raises(GuardRejection):
        guarded_eval('bad == 1', context)
    with pytest.raises(GuardRejection):
        guarded_eval('bad != 1', context)
    with pytest.raises(GuardRejection):
        guarded_eval('1 == bad', context)
    with pytest.raises(GuardRejection):
        guarded_eval('1 != bad', context)
    assert guarded_eval('good == 1', context) is True
    assert guarded_eval('good != 1', context) is False
    assert guarded_eval('1 == good', context) is True
    assert guarded_eval('1 != good', context) is False