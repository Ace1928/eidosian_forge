import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_strict_commit(self):
    """Commit with --strict works if everything is known"""
    ignores._set_user_ignores([])
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a'])
    tree.add('a')
    self.run_bzr('commit --strict -m adding-a', working_dir='tree')