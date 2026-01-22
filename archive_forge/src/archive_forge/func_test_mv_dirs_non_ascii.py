import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_mv_dirs_non_ascii(self):
    """Move directory with non-ascii name and containing files.

        Regression test for bug 185211.
        """
    tree = self.make_branch_and_tree('.')
    self.build_tree(['abc§/', 'abc§/foo'])
    tree.add(['abc§/', 'abc§/foo'])
    tree.commit('checkin')
    tree.rename_one('abc§', 'abc')
    self.run_bzr('ci -m "non-ascii mv"')