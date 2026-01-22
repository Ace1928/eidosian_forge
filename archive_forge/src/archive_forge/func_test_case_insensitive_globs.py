import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_case_insensitive_globs(self):
    if os.path.normcase('AbC') == 'AbC':
        self.skipTest('Test requires case insensitive globbing function')
    self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
    self.assertCommandLine(['A/b.c'], 'A/B*')