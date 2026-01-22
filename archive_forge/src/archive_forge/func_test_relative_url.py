import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_relative_url(self):

    def test(expected, base, other):
        result = urlutils.relative_url(base, other)
        self.assertEqual(expected, result)
    test('a', 'http://host/', 'http://host/a')
    test('http://entirely/different', 'sftp://host/branch', 'http://entirely/different')
    test('../person/feature', 'http://host/branch/mainline', 'http://host/branch/person/feature')
    test('..', 'http://host/branch', 'http://host/')
    test('http://host2/branch', 'http://host1/branch', 'http://host2/branch')
    test('.', 'http://host1/branch', 'http://host1/branch')
    test('../../../branch/2b', 'file:///home/jelmer/foo/bar/2b', 'file:///home/jelmer/branch/2b')
    test('../../branch/2b', 'sftp://host/home/jelmer/bar/2b', 'sftp://host/home/jelmer/branch/2b')
    test('../../branch/feature/%2b', 'http://host/home/jelmer/bar/%2b', 'http://host/home/jelmer/branch/feature/%2b')
    test('../../branch/feature/2b', 'http://host/home/jelmer/bar/2b/', 'http://host/home/jelmer/branch/feature/2b')
    test('../../branch/feature/2b/', 'http://host/home/jelmer/bar/2b/', 'http://host/home/jelmer/branch/feature/2b/')
    test('../../branch/feature/2b/', 'http://host/home/jelmer/bar/2b', 'http://host/home/jelmer/branch/feature/2b/')
    test('http://host/a', 'http://host', 'http://host/a')
    test('http://host/', 'http://host', 'http://host/')
    test('http://host', 'http://host/', 'http://host')
    if sys.platform == 'win32':
        test('../../other/path', 'file:///C:/path/to', 'file:///C:/other/path')
        test('../../other/path', 'file://HOST/base/path/to', 'file://HOST/base/other/path')
        test('file:///D:/other/path', 'file:///C:/path/to', 'file:///D:/other/path')