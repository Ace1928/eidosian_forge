import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_glob_paths(self):
    self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
    self.assertCommandLine(['a/b.c', 'a/c.c'], 'a/*.c')
    self.build_tree(['b/', 'b/b.c', 'b/d.c', 'b/d.h'])
    self.assertCommandLine(['a/b.c', 'b/b.c'], '*/b.c')
    self.assertCommandLine(['a/b.c', 'a/c.c', 'b/b.c', 'b/d.c'], '*/*.c')
    self.assertCommandLine(['*/*.qqq'], '*/*.qqq')