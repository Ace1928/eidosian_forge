import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_strip_trailing_slash(self):
    sts = urlutils.strip_trailing_slash
    if sys.platform == 'win32':
        self.assertEqual('file:///C|/', sts('file:///C|/'))
        self.assertEqual('file:///C:/foo', sts('file:///C:/foo'))
        self.assertEqual('file:///C|/foo', sts('file:///C|/foo/'))
    else:
        self.assertEqual('file:///', sts('file:///'))
        self.assertEqual('file:///foo', sts('file:///foo'))
        self.assertEqual('file:///foo', sts('file:///foo/'))
    self.assertEqual('http://host/', sts('http://host/'))
    self.assertEqual('http://host/foo', sts('http://host/foo'))
    self.assertEqual('http://host/foo', sts('http://host/foo/'))
    self.assertEqual('http://host', sts('http://host'))
    self.assertEqual('file://', sts('file://'))
    self.assertEqual('random+scheme://user:pass@ahost:port/path', sts('random+scheme://user:pass@ahost:port/path'))
    self.assertEqual('random+scheme://user:pass@ahost:port/path', sts('random+scheme://user:pass@ahost:port/path/'))
    self.assertEqual('random+scheme://user:pass@ahost:port/', sts('random+scheme://user:pass@ahost:port/'))
    self.assertEqual('path/to/foo', sts('path/to/foo'))
    self.assertEqual('path/to/foo', sts('path/to/foo/'))
    self.assertEqual('../to/foo', sts('../to/foo/'))
    self.assertEqual('path/../foo', sts('path/../foo/'))