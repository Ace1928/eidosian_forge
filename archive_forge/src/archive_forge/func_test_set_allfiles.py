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
def test_set_allfiles(self):
    file_list = FileList()
    files = ['a', 'b', 'c']
    file_list.set_allfiles(files)
    self.assertEqual(file_list.allfiles, files)