import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_a_text_merge_in_a_checkout(self):
    trunk = self.make_branch_and_tree('trunk')
    u1 = trunk.branch.create_checkout('u1')
    self.build_tree_contents([('u1/hosts', b'initial contents\n')])
    u1.add('hosts')
    self.run_bzr('commit -m add-hosts u1')
    u2 = trunk.branch.create_checkout('u2')
    self.build_tree_contents([('u2/hosts', b'altered in u2\n')])
    self.run_bzr('commit -m checkin-from-u2 u2')
    self.build_tree_contents([('u1/hosts', b'first offline change in u1\n')])
    self.run_bzr('commit -m checkin-offline --local u1')
    self.run_bzr('update u1', retcode=1)
    self.assertFileEqual(b'<<<<<<< TREE\nfirst offline change in u1\n=======\naltered in u2\n>>>>>>> MERGE-SOURCE\n', 'u1/hosts')
    self.run_bzr('resolved u1/hosts')
    self.build_tree_contents([('u1/hosts', b'merge resolution\n')])
    self.run_bzr('commit -m checkin-merge-of-the-offline-work-from-u1 u1')