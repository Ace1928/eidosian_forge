import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_unquote_to_bytes(self):
    self.assertEqual(b'%', urlutils.unquote_to_bytes('%25'))
    self.assertEqual(b'\xc3\xa5', urlutils.unquote_to_bytes('%C3%A5'))