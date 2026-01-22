import os
import re
import unittest
from distutils import debug
from distutils.log import WARN
from distutils.errors import DistutilsTemplateError
from distutils.filelist import glob_to_re, translate_pattern, FileList
from distutils import filelist
from test.support import os_helper
from test.support import captured_stdout
from distutils.tests import support
@os_helper.skip_unless_symlink
def test_missing_symlink(self):
    with os_helper.temp_cwd():
        os.symlink('foo', 'bar')
        self.assertEqual(filelist.findall(), [])