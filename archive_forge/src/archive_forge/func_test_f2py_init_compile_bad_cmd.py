import sys
import os
import uuid
from importlib import import_module
import pytest
import numpy.f2py
from . import util
def test_f2py_init_compile_bad_cmd():
    try:
        temp = sys.executable
        sys.executable = 'does not exist'
        ret_val = numpy.f2py.compile(b'invalid')
        assert ret_val == 127
    finally:
        sys.executable = temp