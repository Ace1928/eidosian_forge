import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_with_pdb(self):
    """Check stripping Python arguments before bzr script per lp:587868"""
    self.assertCommandLine(['rocks'], '-m pdb rocks', ['rocks'])
    self.build_tree(['d/', 'd/f1', 'd/f2'])
    self.assertCommandLine(['rm', 'x*'], '-m pdb rm x*', ['rm', 'x*'])
    self.assertCommandLine(['add', 'd/f1', 'd/f2'], '-m pdb add d/*', ['add', 'd/*'])