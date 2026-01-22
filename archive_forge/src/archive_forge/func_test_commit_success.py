import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_success(self):
    """Successful commit should not leave behind a bzr-commit-* file"""
    self.make_branch_and_tree('.')
    self.run_bzr('commit --unchanged -m message')
    self.assertEqual('', self.run_bzr('unknowns')[0])
    self.run_bzr(['commit', '--unchanged', '-m', 'fooµ'])
    self.assertEqual('', self.run_bzr('unknowns')[0])