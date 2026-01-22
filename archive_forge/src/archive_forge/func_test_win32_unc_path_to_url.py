import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_win32_unc_path_to_url(self):
    self.requireFeature(features.win32_feature)
    to_url = urlutils._win32_local_path_to_url
    self.assertEqual('file://HOST/path', to_url('\\\\HOST\\path'))
    self.assertEqual('file://HOST/path', to_url('//HOST/path'))
    try:
        result = to_url('//HOST/path/to/räksmörgås')
    except UnicodeError:
        raise TestSkipped('local encoding cannot handle unicode')
    self.assertEqual('file://HOST/path/to/r%C3%A4ksm%C3%B6rg%C3%A5s', result)
    self.assertFalse(isinstance(result, str))