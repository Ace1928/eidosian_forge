import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_verbose_commit_with_unchanged(self):
    """Unchanged files should not be listed by default in verbose output"""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['hello.txt', 'unchanged.txt'])
    tree.add('unchanged.txt')
    self.run_bzr('commit -m unchanged unchanged.txt')
    tree.add('hello.txt')
    out, err = self.run_bzr('commit -m added')
    self.assertEqual('', out)
    self.assertContainsRe(err, '^Committing to: .*\nadded hello\\.txt\nCommitted revision 2\\.$\n')