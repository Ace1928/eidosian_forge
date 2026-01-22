import os
import sys
import shutil
import tempfile
import unittest
import sysconfig
from copy import deepcopy
from test.support import os_helper
from distutils import log
from distutils.log import DEBUG, INFO, WARN, ERROR, FATAL
from distutils.core import Distribution
def fixup_build_ext(cmd):
    """Function needed to make build_ext tests pass.

    When Python was built with --enable-shared on Unix, -L. is not enough to
    find libpython<blah>.so, because regrtest runs in a tempdir, not in the
    source directory where the .so lives.

    When Python was built with in debug mode on Windows, build_ext commands
    need their debug attribute set, and it is not done automatically for
    some reason.

    This function handles both of these things.  Example use:

        cmd = build_ext(dist)
        support.fixup_build_ext(cmd)
        cmd.ensure_finalized()

    Unlike most other Unix platforms, Mac OS X embeds absolute paths
    to shared libraries into executables, so the fixup is not needed there.
    """
    if os.name == 'nt':
        cmd.debug = sys.executable.endswith('_d.exe')
    elif sysconfig.get_config_var('Py_ENABLE_SHARED'):
        runshared = sysconfig.get_config_var('RUNSHARED')
        if runshared is None:
            cmd.library_dirs = ['.']
        elif sys.platform == 'darwin':
            cmd.library_dirs = []
        else:
            name, equals, value = runshared.partition('=')
            cmd.library_dirs = [d for d in value.split(os.pathsep) if d]