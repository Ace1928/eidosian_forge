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
def check_decl(self, symbol, headers=None, include_dirs=None):
    self._check_compiler()
    body = textwrap.dedent('\n            int main(void)\n            {\n            #ifndef %s\n                (void) %s;\n            #endif\n                ;\n                return 0;\n            }') % (symbol, symbol)
    return self.try_compile(body, headers, include_dirs)