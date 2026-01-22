import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion
def get_libgfortran_dir(self):
    if sys.platform[:5] == 'linux':
        libgfortran_name = 'libgfortran.so'
    elif sys.platform == 'darwin':
        libgfortran_name = 'libgfortran.dylib'
    else:
        libgfortran_name = None
    libgfortran_dir = None
    if libgfortran_name:
        find_lib_arg = ['-print-file-name={0}'.format(libgfortran_name)]
        try:
            output = subprocess.check_output(self.compiler_f77 + find_lib_arg)
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            output = filepath_from_subprocess_output(output)
            libgfortran_dir = os.path.dirname(output)
    return libgfortran_dir