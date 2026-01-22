import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_pending_merge(self):
    wt = self.create_simple_tree()
    tree2 = wt.controldir.sprout('tree2').open_workingtree()
    tree2.commit('unchanged', rev_id=b'b3')
    wt.branch.fetch(tree2.branch)
    wt.set_pending_merges([b'b3'])
    os.chdir('tree')
    out, err = self.run_bzr('uncommit --force')
    self.assertEqual([b'a1', b'b3'], wt.get_parent_ids())