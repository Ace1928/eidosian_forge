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
def test_find_file_decorated1():

    @decorator
    def noop1(f):

        def wrapper(*a, **kw):
            return f(*a, **kw)
        return wrapper

    @noop1
    def f(x):
        """My docstring"""
    match_pyfiles(oinspect.find_file(f), os.path.abspath(__file__))
    assert f.__doc__ == 'My docstring'