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
@unittest.skipIf(is_wasi, 'Incompatible with WASI mapdir and OOT builds')
def test_get_config_h_filename(self):
    config_h = sysconfig.get_config_h_filename()
    self.assertTrue(os.path.isfile(config_h), config_h)