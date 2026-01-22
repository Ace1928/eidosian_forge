from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_add_second_file(self):
    builder = self.build_a_rev()
    rev_id2 = builder.build_snapshot(None, [('add', ('b', b'b-id', 'file', b'content_b'))], revision_id=b'B-id')
    self.assertEqual(b'B-id', rev_id2)
    branch = builder.get_branch()
    self.assertEqual((2, rev_id2), branch.last_revision_info())
    rev_tree = branch.repository.revision_tree(rev_id2)
    rev_tree.lock_read()
    self.addCleanup(rev_tree.unlock)
    self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file'), ('b', b'b-id', 'file')], rev_tree)
    self.assertEqual(b'content_b', rev_tree.get_file_text('b'))