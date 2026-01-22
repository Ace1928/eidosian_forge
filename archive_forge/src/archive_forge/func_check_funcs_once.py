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
def check_funcs_once(self, funcs, headers=None, include_dirs=None, libraries=None, library_dirs=None, decl=False, call=False, call_args=None):
    """Check a list of functions at once.

        This is useful to speed up things, since all the functions in the funcs
        list will be put in one compilation unit.

        Arguments
        ---------
        funcs : seq
            list of functions to test
        include_dirs : seq
            list of header paths
        libraries : seq
            list of libraries to link the code snippet to
        library_dirs : seq
            list of library paths
        decl : dict
            for every (key, value), the declaration in the value will be
            used for function in key. If a function is not in the
            dictionary, no declaration will be used.
        call : dict
            for every item (f, value), if the value is True, a call will be
            done to the function f.
        """
    self._check_compiler()
    body = []
    if decl:
        for f, v in decl.items():
            if v:
                body.append('int %s (void);' % f)
    body.append('#ifdef _MSC_VER')
    for func in funcs:
        body.append('#pragma function(%s)' % func)
    body.append('#endif')
    body.append('int main (void) {')
    if call:
        for f in funcs:
            if f in call and call[f]:
                if not (call_args and f in call_args and call_args[f]):
                    args = ''
                else:
                    args = call_args[f]
                body.append('  %s(%s);' % (f, args))
            else:
                body.append('  %s;' % f)
    else:
        for f in funcs:
            body.append('  %s;' % f)
    body.append('  return 0;')
    body.append('}')
    body = '\n'.join(body) + '\n'
    return self.try_link(body, headers, include_dirs, libraries, library_dirs)