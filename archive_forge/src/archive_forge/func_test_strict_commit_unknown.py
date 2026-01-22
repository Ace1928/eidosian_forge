import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_strict_commit_unknown(self):
    """commit --strict fails if a file is unknown"""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a'])
    tree.add('a')
    tree.commit('adding a')
    self.build_tree(['tree/b', 'tree/c'])
    tree.add('b')
    self.run_bzr_error(['Commit refused because there are unknown files'], 'commit --strict -m add-b', working_dir='tree')
    self.run_bzr('commit --strict -m add-b --no-strict', working_dir='tree')