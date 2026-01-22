import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_parent_win32(self):
    self._with_win32_paths()
    self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///A:/b', 'file:///A:/')
    self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///A:/b/c', 'file:///A:/b')