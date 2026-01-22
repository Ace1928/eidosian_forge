import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_basename(self):
    basename = urlutils.basename
    if sys.platform == 'win32':
        self.assertRaises(urlutils.InvalidURL, basename, 'file:///path/to/foo')
        self.assertEqual('foo', basename('file:///C|/foo'))
        self.assertEqual('foo', basename('file:///C:/foo'))
        self.assertEqual('', basename('file:///C:/'))
    else:
        self.assertEqual('foo', basename('file:///foo'))
        self.assertEqual('', basename('file:///'))
    self.assertEqual('foo', basename('http://host/path/to/foo'))
    self.assertEqual('foo', basename('http://host/path/to/foo/'))
    self.assertEqual('', basename('http://host/path/to/foo/', exclude_trailing_slash=False))
    self.assertEqual('path', basename('http://host/path'))
    self.assertEqual('', basename('http://host/'))
    self.assertEqual('', basename('http://host'))
    self.assertEqual('path', basename('http:///nohost/path'))
    self.assertEqual('path', basename('random+scheme://user:pass@ahost:port/path'))
    self.assertEqual('path', basename('random+scheme://user:pass@ahost:port/path/'))
    self.assertEqual('', basename('random+scheme://user:pass@ahost:port/'))
    self.assertEqual('foo', basename('path/to/foo'))
    self.assertEqual('foo', basename('path/to/foo/'))
    self.assertEqual('', basename('path/to/foo/', exclude_trailing_slash=False))
    self.assertEqual('foo', basename('path/../foo'))
    self.assertEqual('foo', basename('../path/foo'))