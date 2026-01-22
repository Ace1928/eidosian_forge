import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_escape_tildes(self):
    self.assertEqual('~foo', urlutils.escape('~foo'))