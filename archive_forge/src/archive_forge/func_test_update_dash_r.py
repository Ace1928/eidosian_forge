import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_dash_r(self):
    master = self.make_branch_and_tree('master')
    os.chdir('master')
    self.build_tree(['./file1'])
    master.add(['file1'])
    master.commit('one', rev_id=b'm1')
    self.build_tree(['./file2'])
    master.add(['file2'])
    master.commit('two', rev_id=b'm2')
    sr = ScriptRunner()
    sr.run_script(self, '\n$ brz update -r 1\n2>-D  file2\n2>All changes applied successfully.\n2>Updated to revision 1 of .../master\n')
    self.assertPathExists('./file1')
    self.assertPathDoesNotExist('./file2')
    self.assertEqual([b'm1'], master.get_parent_ids())