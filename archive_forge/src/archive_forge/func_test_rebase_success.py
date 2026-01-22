import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_rebase_success(self):
    self.assertEqual('../bar', urlutils.rebase_url('bar', 'http://baz/', 'http://baz/qux'))
    self.assertEqual('qux/bar', urlutils.rebase_url('bar', 'http://baz/qux', 'http://baz/'))
    self.assertEqual('.', urlutils.rebase_url('foo', 'http://bar/', 'http://bar/foo/'))
    self.assertEqual('qux/bar', urlutils.rebase_url('../bar', 'http://baz/qux/foo', 'http://baz/'))