from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_named_tuple():

    class GoodNamedTuple(NamedTuple):
        a: str
        pass

    class BadNamedTuple(NamedTuple):
        a: str

        def __getitem__(self, key):
            return None
    good = GoodNamedTuple(a='x')
    bad = BadNamedTuple(a='x')
    context = limited(data=good)
    assert guarded_eval('data[0]', context) == 'x'
    context = limited(data=bad)
    with pytest.raises(GuardRejection):
        guarded_eval('data[0]', context)