import os
from breezy import branch as _mod_branch
from breezy import errors, osutils
from breezy import revision as _mod_revision
from breezy import tests, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import remote
from breezy.tests import features
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_branch_parent(self):
    source = self.make_branch('source')
    target = source.controldir.sprout(self.get_url('target')).open_branch()
    self.assertEqual(urlutils.strip_segment_parameters(source.user_url), urlutils.strip_segment_parameters(target.get_parent()))