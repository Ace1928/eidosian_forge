import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_case_insensitive_globbing(self):
    if os.path.normcase('AbC') == 'AbC':
        self.skipTest('Test requires case insensitive globbing function')
    self.build_ascii_tree()
    self._run_testset([[['A'], ['A']], [['A?'], ['a1', 'a2']]])