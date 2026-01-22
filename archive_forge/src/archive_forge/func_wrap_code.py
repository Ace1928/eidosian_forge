import sys
import os
import shutil
import tempfile
from subprocess import STDOUT, CalledProcessError, check_output
from string import Template
from warnings import warn
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.utilities.codegen import (make_routine, get_code_generator,
from sympy.utilities.iterables import iterable
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.decorator import doctest_depends_on
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools.extension import Extension
from setuptools import setup
from numpy import get_include
def wrap_code(self, routines, helpers=None):
    helpers = helpers if helpers is not None else []
    funcname = 'wrapped_' + str(id(routines) + id(helpers))
    workdir = self.filepath or tempfile.mkdtemp('_sympy_compile')
    if not os.access(workdir, os.F_OK):
        os.mkdir(workdir)
    oldwork = os.getcwd()
    os.chdir(workdir)
    try:
        sys.path.append(workdir)
        self._generate_code(routines, helpers)
        self._prepare_files(routines, funcname)
        self._process_files(routines)
        mod = __import__(self.module_name)
    finally:
        sys.path.remove(workdir)
        CodeWrapper._module_counter += 1
        os.chdir(oldwork)
        if not self.filepath:
            try:
                shutil.rmtree(workdir)
            except OSError:
                pass
    return self._get_wrapped_function(mod, funcname)