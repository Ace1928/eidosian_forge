import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_other_branch_commit(self):
    outer_tree = self.make_branch_and_tree('.')
    inner_tree = self.make_branch_and_tree('branch')
    self.build_tree_contents([('branch/foo.c', b'int main() {}'), ('branch/bar.c', b'int main() {}')])
    inner_tree.add(['foo.c', 'bar.c'])
    self.run_bzr('commit -m newstuff branch/foo.c .', retcode=3)
    self.run_bzr('commit -m newstuff branch/foo.c')
    self.run_bzr('commit -m newstuff branch')
    self.run_bzr_error(['No changes to commit'], 'commit -m newstuff branch')