from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_tags_with_mainline_ghosts(self):
    tree = self.make_branch_and_tree('tree1')
    tree.set_parent_ids([b'spooky'], allow_leftmost_as_ghost=True)
    tree.add('')
    tree.commit('msg1', rev_id=b'rev1')
    tree.commit('msg2', rev_id=b'rev2')
    tree.branch.tags.set_tag('unknown', b'out-of-mainline')
    tree.branch.tags.set_tag('ghost', b'spooky')
    tree.branch.tags.set_tag('tag1', b'rev1')
    tree.branch.tags.set_tag('tag2', b'rev2')
    out, err = self.run_bzr('tags -d tree1', encoding='utf-8')
    self.assertEqual(out, 'ghost                ?\ntag1                 1\ntag2                 2\nunknown              ?\n')
    self.assertEqual('', err)