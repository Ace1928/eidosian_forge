import contextlib
import os
import shutil
import subprocess
import sys
import textwrap
import unittest
from distutils import sysconfig
from distutils.ccompiler import get_default_compiler
from distutils.tests import support
from test.support import swap_item, requires_subprocess, is_wasi
from test.support.os_helper import TESTFN
from test.support.warnings_helper import check_warnings
@requires_subprocess()
def test_customize_compiler_before_get_config_vars(self):
    with open(TESTFN, 'w') as f:
        f.writelines(textwrap.dedent("                from distutils.core import Distribution\n                config = Distribution().get_command_obj('config')\n                # try_compile may pass or it may fail if no compiler\n                # is found but it should not raise an exception.\n                rc = config.try_compile('int x;')\n                "))
    p = subprocess.Popen([str(sys.executable), TESTFN], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    outs, errs = p.communicate()
    self.assertEqual(0, p.returncode, 'Subprocess failed: ' + outs)