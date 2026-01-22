from __future__ import annotations
import datetime
import functools
import operator
import pickle
from array import array
import pytest
from tlz import curry
from dask import get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable
from dask.utils import (
from dask.utils_test import inc
def test_dispatch_lazy():
    foo = Dispatch()
    foo.register(int, lambda a: a)
    import decimal

    def foo_dec(a):
        return a + 1

    @foo.register_lazy('decimal')
    def register_decimal():
        import decimal
        foo.register(decimal.Decimal, foo_dec)
    assert foo.dispatch(decimal.Decimal) == foo_dec
    assert foo(decimal.Decimal(1)) == decimal.Decimal(2)
    assert foo(1) == 1