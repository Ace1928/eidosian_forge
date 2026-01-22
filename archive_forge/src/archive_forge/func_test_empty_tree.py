import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_empty_tree(self):
    self.build_tree([])
    self._run_testset([[['a'], ['a']], [['?'], ['?']], [['*'], ['*']], [['a', 'a'], ['a', 'a']]])