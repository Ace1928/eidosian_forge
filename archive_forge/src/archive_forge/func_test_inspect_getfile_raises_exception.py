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
def test_inspect_getfile_raises_exception():
    """Check oinspect.find_file/getsource/find_source_lines expectations"""
    with pytest.raises(TypeError):
        inspect.getfile(type)
    with pytest.raises(OSError):
        inspect.getfile(SourceModuleMainTest)