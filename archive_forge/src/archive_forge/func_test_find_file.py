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
def test_find_file():
    match_pyfiles(oinspect.find_file(test_find_file), os.path.abspath(__file__))
    assert oinspect.find_file(type) is None
    assert oinspect.find_file(SourceModuleMainTest) is None
    assert oinspect.find_file(SourceModuleMainTest()) is None