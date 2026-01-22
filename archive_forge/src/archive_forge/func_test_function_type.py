import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_function_type(self):
    if sys.platform == 'win32':
        self.assertEqual(urlutils._win32_local_path_to_url, urlutils.local_path_to_url)
        self.assertEqual(urlutils._win32_local_path_from_url, urlutils.local_path_from_url)
    else:
        self.assertEqual(urlutils._posix_local_path_to_url, urlutils.local_path_to_url)
        self.assertEqual(urlutils._posix_local_path_from_url, urlutils.local_path_from_url)