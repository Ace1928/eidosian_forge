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
def is_free_format(file):
    """Check if file is in free format Fortran."""
    result = 0
    with open(file, encoding='latin1') as f:
        line = f.readline()
        n = 10000
        if _has_f_header(line) or _has_fix_header(line):
            n = 0
        elif _has_f90_header(line):
            n = 0
            result = 1
        while n > 0 and line:
            line = line.rstrip()
            if line and line[0] != '!':
                n -= 1
                if line[0] != '\t' and _free_f90_start(line[:5]) or line[-1:] == '&':
                    result = 1
                    break
            line = f.readline()
    return result