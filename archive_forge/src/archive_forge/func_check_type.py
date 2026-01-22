import os
import signal
import subprocess
import sys
import textwrap
import warnings
from distutils.command.config import config as old_config
from distutils.command.config import LANG_EXT
from distutils import log
from distutils.file_util import copy_file
from distutils.ccompiler import CompileError, LinkError
import distutils
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.mingw32ccompiler import generate_manifest
from numpy.distutils.command.autodist import (check_gcc_function_attribute,
def check_type(self, type_name, headers=None, include_dirs=None, library_dirs=None):
    """Check type availability. Return True if the type can be compiled,
        False otherwise"""
    self._check_compiler()
    body = textwrap.dedent('\n            int main(void) {\n              if ((%(name)s *) 0)\n                return 0;\n              if (sizeof (%(name)s))\n                return 0;\n            }\n            ') % {'name': type_name}
    st = False
    try:
        try:
            self._compile(body % {'type': type_name}, headers, include_dirs, 'c')
            st = True
        except distutils.errors.CompileError:
            st = False
    finally:
        self._clean()
    return st