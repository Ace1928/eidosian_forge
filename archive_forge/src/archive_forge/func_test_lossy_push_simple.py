from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
def test_lossy_push_simple(self):
    source_tree = self.make_branch_and_tree('source')
    self.build_tree(['source/a', 'source/b'])
    source_tree.add(['a', 'b'])
    revid1 = source_tree.commit('msg')
    target_tree = self.make_branch_and_tree('target', format=DummyForeignVcsDirFormat())
    target_tree.branch.lock_write()
    try:
        pushresult = source_tree.branch.push(target_tree.branch, lossy=True)
    finally:
        target_tree.branch.unlock()
    self.assertEqual(revision.NULL_REVISION, pushresult.old_revid)
    self.assertEqual({revid1: target_tree.branch.last_revision()}, pushresult.revidmap)
    self.assertEqual(pushresult.revidmap[revid1], pushresult.new_revid)