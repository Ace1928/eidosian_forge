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
def test_find_file_decorated2():

    @decorator
    def noop2(f, *a, **kw):
        return f(*a, **kw)

    @noop2
    @noop2
    @noop2
    def f(x):
        """My docstring 2"""
    match_pyfiles(oinspect.find_file(f), os.path.abspath(__file__))
    assert f.__doc__ == 'My docstring 2'