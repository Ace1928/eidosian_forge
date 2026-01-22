import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_bound_local(self):
    t_a = self.make_branch_and_tree('a')
    rev_id1 = t_a.commit('commit 1')
    rev_id2 = t_a.commit('commit 2')
    rev_id3 = t_a.commit('commit 3')
    b = t_a.branch.create_checkout('b').branch
    out, err = self.run_bzr(['uncommit', '--local', 'b', '--force'])
    self.assertEqual(rev_id3, t_a.last_revision())
    self.assertEqual((3, rev_id3), t_a.branch.last_revision_info())
    self.assertEqual((2, rev_id2), b.last_revision_info())