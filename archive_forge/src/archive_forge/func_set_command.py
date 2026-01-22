import os
import sys
import re
from pathlib import Path
from distutils.sysconfig import get_python_lib
from distutils.fancy_getopt import FancyGetopt
from distutils.errors import DistutilsModuleError, \
from distutils.util import split_quoted, strtobool
from numpy.distutils.ccompiler import CCompiler, gen_lib_options
from numpy.distutils import log
from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
from numpy.distutils.exec_command import find_executable
from numpy.distutils import _shell_utils
from .environment import EnvironmentConfig
def set_command(self, key, value):
    if not key in self._executable_keys:
        raise ValueError("unknown executable '%s' for class %s" % (key, self.__class__.__name__))
    if is_string(value):
        value = split_quoted(value)
    assert value is None or is_sequence_of_strings(value[1:]), (key, value)
    self.executables[key] = value