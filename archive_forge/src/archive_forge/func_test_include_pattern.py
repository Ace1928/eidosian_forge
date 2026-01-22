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
def test_include_pattern(self):
    file_list = FileList()
    file_list.set_allfiles([])
    self.assertFalse(file_list.include_pattern('*.py'))
    file_list = FileList()
    file_list.set_allfiles(['a.py', 'b.txt'])
    self.assertTrue(file_list.include_pattern('*.py'))
    file_list = FileList()
    self.assertIsNone(file_list.allfiles)
    file_list.set_allfiles(['a.py', 'b.txt'])
    file_list.include_pattern('*')
    self.assertEqual(file_list.allfiles, ['a.py', 'b.txt'])