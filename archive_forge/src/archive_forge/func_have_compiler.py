import os
import shutil
import pytest
from tempfile import mkstemp, mkdtemp
from subprocess import Popen, PIPE
import importlib.metadata
from distutils.errors import DistutilsError
from numpy.testing import assert_, assert_equal, assert_raises
from numpy.distutils import ccompiler, customized_ccompiler
from numpy.distutils.system_info import system_info, ConfigParser, mkl_info
from numpy.distutils.system_info import AliasedOptionError
from numpy.distutils.system_info import default_lib_dirs, default_include_dirs
from numpy.distutils import _shell_utils
def have_compiler():
    """ Return True if there appears to be an executable compiler
    """
    compiler = customized_ccompiler()
    try:
        cmd = compiler.compiler
    except AttributeError:
        try:
            if not compiler.initialized:
                compiler.initialize()
        except (DistutilsError, ValueError):
            return False
        cmd = [compiler.cc]
    try:
        p = Popen(cmd, stdout=PIPE, stderr=PIPE)
        p.stdout.close()
        p.stderr.close()
        p.wait()
    except OSError:
        return False
    return True