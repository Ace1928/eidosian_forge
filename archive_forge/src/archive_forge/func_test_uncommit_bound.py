import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_bound(self):
    os.mkdir('a')
    a = BzrDirMetaFormat1().initialize('a')
    a.create_repository()
    a.create_branch()
    t_a = a.create_workingtree()
    t_a.commit('commit 1')
    t_a.commit('commit 2')
    t_a.commit('commit 3')
    b = t_a.branch.create_checkout('b').branch
    uncommit.uncommit(b)
    self.assertEqual(b.last_revision_info()[0], 2)
    self.assertEqual(t_a.branch.last_revision_info()[0], 2)
    t_a.update()
    t_a.commit('commit 3b')
    self.assertRaises(BoundBranchOutOfDate, uncommit.uncommit, b)
    b.pull(t_a.branch)
    uncommit.uncommit(b)