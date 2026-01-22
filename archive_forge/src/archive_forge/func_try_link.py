import os, re
from distutils.core import Command
from distutils.errors import DistutilsExecError
from distutils.sysconfig import customize_compiler
from distutils import log
def try_link(self, body, headers=None, include_dirs=None, libraries=None, library_dirs=None, lang='c'):
    """Try to compile and link a source file, built from 'body' and
        'headers', to executable form.  Return true on success, false
        otherwise.
        """
    from distutils.ccompiler import CompileError, LinkError
    self._check_compiler()
    try:
        self._link(body, headers, include_dirs, libraries, library_dirs, lang)
        ok = True
    except (CompileError, LinkError):
        ok = False
    log.info(ok and 'success!' or 'failure.')
    self._clean()
    return ok