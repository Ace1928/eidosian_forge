import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_ipv6(self):
    parsed = urlutils.URL.from_string('http://[1:2:3::40]/one')
    self.assertEqual('http', parsed.scheme)
    self.assertIs(None, parsed.port)
    self.assertIs(None, parsed.user)
    self.assertIs(None, parsed.password)
    self.assertEqual('1:2:3::40', parsed.host)
    self.assertEqual('/one', parsed.path)