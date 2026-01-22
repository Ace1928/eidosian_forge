import os
import tarfile
import unittest
import warnings
import zipfile
from os.path import join
from textwrap import dedent
from test.support import captured_stdout
from test.support.warnings_helper import check_warnings
from distutils.command.sdist import sdist, show_formats
from distutils.core import Distribution
from distutils.tests.test_config import BasePyPIRCCommandTestCase
from distutils.errors import DistutilsOptionError
from distutils.spawn import find_executable
from distutils.log import WARN
from distutils.filelist import FileList
from distutils.archive_util import ARCHIVE_FORMATS
from distutils.core import setup
import somecode
@unittest.skipUnless(ZLIB_SUPPORT, 'Need zlib support to run')
def test_metadata_check_option(self):
    dist, cmd = self.get_cmd(metadata={})
    cmd.ensure_finalized()
    cmd.run()
    warnings = [msg for msg in self.get_logs(WARN) if msg.startswith('warning: check:')]
    self.assertEqual(len(warnings), 2)
    self.clear_logs()
    dist, cmd = self.get_cmd()
    cmd.ensure_finalized()
    cmd.metadata_check = 0
    cmd.run()
    warnings = [msg for msg in self.get_logs(WARN) if msg.startswith('warning: check:')]
    self.assertEqual(len(warnings), 0)