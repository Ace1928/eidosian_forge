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
@unittest.skipIf(sysconfig.get_config_var('CUSTOMIZED_OSX_COMPILER'), 'compiler flags customized')
def test_sysconfig_compiler_vars(self):
    import sysconfig as global_sysconfig
    if sysconfig.get_config_var('CUSTOMIZED_OSX_COMPILER'):
        self.skipTest('compiler flags customized')
    self.assertEqual(global_sysconfig.get_config_var('LDSHARED'), sysconfig.get_config_var('LDSHARED'))
    self.assertEqual(global_sysconfig.get_config_var('CC'), sysconfig.get_config_var('CC'))