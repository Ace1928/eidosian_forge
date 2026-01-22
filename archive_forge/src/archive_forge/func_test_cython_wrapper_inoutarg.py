import os
import tempfile
import shutil
from io import StringIO
from sympy.core import symbols, Eq
from sympy.utilities.autowrap import (autowrap, binary_function,
from sympy.utilities.codegen import (
from sympy.testing.pytest import raises
from sympy.testing.tmpfiles import TmpFileManager
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np
def test_cython_wrapper_inoutarg():
    from sympy.core.relational import Equality
    x, y, z = symbols('x,y,z')
    code_gen = CythonCodeWrapper(C99CodeGen())
    routine = make_routine('test', Equality(z, x + y + z))
    source = get_string(code_gen.dump_pyx, [routine])
    expected = "cdef extern from 'file.h':\n    void test(double x, double y, double *z)\n\ndef test_c(double x, double y, double z):\n\n    test(x, y, &z)\n    return z"
    assert source == expected