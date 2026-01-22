import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_lossy_foreign(self):
    test_foreign.register_dummy_foreign_for_test(self)
    self.make_branch_and_tree('.', format=test_foreign.DummyForeignVcsDirFormat())
    self.run_bzr('commit --lossy --unchanged -m message')
    output = self.run_bzr('revision-info')[0]
    self.assertTrue(output.startswith('1 dummy-'))