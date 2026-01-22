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
def test_find_source_lines():
    assert oinspect.find_source_lines(test_find_source_lines) == THIS_LINE_NUMBER + 3
    assert oinspect.find_source_lines(type) is None
    assert oinspect.find_source_lines(SourceModuleMainTest) is None
    assert oinspect.find_source_lines(SourceModuleMainTest()) is None