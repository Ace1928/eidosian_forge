import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_tree_ascii(self):
    """Checks the glob expansion and path separation char
        normalization"""
    self.build_ascii_tree()
    self._run_testset([[['a'], ['a']], [['a', 'a'], ['a', 'a']], [['d'], ['d']], [['d/'], ['d/']], [['a*'], ['a', 'a1', 'a2', 'a11', 'a.1']], [['?'], ['a', 'b', 'c', 'd']], [['a?'], ['a1', 'a2']], [['a??'], ['a11', 'a.1']], [['b[1-2]'], ['b1', 'b2']], [['d/*'], ['d/d1', 'd/d2', 'd/e']], [['?/*'], ['c/c1', 'c/c2', 'd/d1', 'd/d2', 'd/e']], [['*/*'], ['c/c1', 'c/c2', 'd/d1', 'd/d2', 'd/e']], [['*/'], ['c/', 'd/']]])