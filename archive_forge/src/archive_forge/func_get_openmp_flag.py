import os
import sys
import textwrap
import warnings
from .pre_build_helpers import compile_test_program
def get_openmp_flag():
    if sys.platform == 'win32':
        return ['/openmp']
    elif sys.platform == 'darwin' and 'openmp' in os.getenv('CPPFLAGS', ''):
        return []
    return ['-fopenmp']