import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_quote_tildes(self):
    if sys.version_info[:2] >= (3, 7):
        self.assertEqual('~foo', urlutils.quote('~foo'))
    else:
        self.assertEqual('%7Efoo', urlutils.quote('~foo'))
    self.assertEqual('~foo', urlutils.quote('~foo', safe='/~'))