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
@pytest.mark.parametrize('decorator', [property, functools.cached_property], ids=['@property', '@cached_property'])
def test_derived_from_prop_cached_prop(decorator):

    class Base:

        @decorator
        def prop(self):
            """A property

            Long details"""
            return 1

    class Derived:

        @decorator
        @derived_from(Base)
        def prop(self):
            """Some extra doc"""
            return 3
    docstring = Derived.prop.__doc__
    assert docstring is not None
    assert docstring.strip().startswith('A property')
    assert any(('inconsistencies' in line for line in docstring.split('\n')))
    assert any(('Some extra doc' in line for line in docstring.split('\n')))