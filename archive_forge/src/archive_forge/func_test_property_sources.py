from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
def test_property_sources():

    def simple_add(a, b):
        """Adds two numbers"""
        return a + b

    class A(object):

        @property
        def foo(self):
            return 'bar'
        foo = foo.setter(lambda self, v: setattr(self, 'bar', v))
        dname = property(oinspect.getdoc)
        adder = property(simple_add)
    i = inspector.info(A.foo, detail_level=1)
    assert 'def foo(self):' in i['source']
    assert 'lambda self, v:' in i['source']
    i = inspector.info(A.dname, detail_level=1)
    assert 'def getdoc(obj)' in i['source']
    i = inspector.info(A.adder, detail_level=1)
    assert 'def simple_add(a, b)' in i['source']