import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_with_merges(self):
    master = self.make_branch_and_tree('master')
    self.build_tree(['master/file'])
    master.add(['file'])
    master.commit('one', rev_id=b'm1')
    self.build_tree(['checkout1/'])
    checkout_dir = bzrdir.BzrDirMetaFormat1().initialize('checkout1')
    checkout_dir.set_branch_reference(master.branch)
    checkout1 = checkout_dir.create_workingtree(b'm1')
    other = master.controldir.sprout('other').open_workingtree()
    self.build_tree(['other/file2'])
    other.add(['file2'])
    other.commit('other2', rev_id=b'o2')
    self.build_tree(['master/file3'])
    master.add(['file3'])
    master.commit('f3', rev_id=b'm2')
    os.chdir('checkout1')
    self.run_bzr('merge ../other')
    self.assertEqual([b'o2'], checkout1.get_parent_ids()[1:])
    self.run_bzr_error(["please run 'brz update'"], 'commit -m merged')
    out, err = self.run_bzr('update')
    self.assertEqual('', out)
    self.assertEqualDiff('+N  file3\nAll changes applied successfully.\nUpdated to revision 2 of branch %s\n' % osutils.pathjoin(self.test_dir, 'master'), err)
    self.assertEqual([b'o2'], checkout1.get_parent_ids()[1:])