import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_unescape(self):
    self.assertEqual('%', urlutils.unescape('%25'))
    self.assertEqual('å', urlutils.unescape('%C3%A5'))
    self.assertEqual('å', urlutils.unescape('%C3%A5'))