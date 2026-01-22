import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_win32_local_path_from_url(self):
    from_url = urlutils._win32_local_path_from_url
    self.assertEqual('C:/path/to/foo', from_url('file:///C|/path/to/foo'))
    self.assertEqual('D:/path/to/räksmörgås', from_url('file:///d|/path/to/r%C3%A4ksm%C3%B6rg%C3%A5s'))
    self.assertEqual('D:/path/to/räksmörgås', from_url('file:///d:/path/to/r%c3%a4ksm%c3%b6rg%c3%a5s'))
    self.assertEqual('/', from_url('file:///'))
    self.assertEqual('C:/path/to/foo', from_url('file:///C|/path/to/foo,branch=foo'))
    self.assertRaises(urlutils.InvalidURL, from_url, 'file:///C:')
    self.assertRaises(urlutils.InvalidURL, from_url, 'file:///c')
    self.assertRaises(urlutils.InvalidURL, from_url, '/path/to/foo')
    self.assertRaises(urlutils.InvalidURL, from_url, 'file:///path/to/foo')