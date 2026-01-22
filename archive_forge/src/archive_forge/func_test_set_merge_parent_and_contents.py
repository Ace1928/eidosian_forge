from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_set_merge_parent_and_contents(self):
    builder = self.build_a_rev()
    builder.start_series()
    self.addCleanup(builder.finish_series)
    builder.build_snapshot([b'A-id'], [('add', ('b', b'b-id', 'file', b'b\ncontent\n'))], revision_id=b'B-id')
    builder.build_snapshot([b'A-id'], [('add', ('c', b'c-id', 'file', b'alt\ncontent\n'))], revision_id=b'C-id')
    builder.build_snapshot([b'B-id', b'C-id'], [('add', ('c', b'c-id', 'file', b'alt\ncontent\n'))], revision_id=b'D-id')
    repo = builder.get_branch().repository
    self.assertEqual({b'B-id': (b'A-id',), b'C-id': (b'A-id',), b'D-id': (b'B-id', b'C-id')}, repo.get_parent_map([b'B-id', b'C-id', b'D-id']))
    d_tree = repo.revision_tree(b'D-id')
    self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file'), ('b', b'b-id', 'file'), ('c', b'c-id', 'file')], d_tree)
    self.assertEqual(b'C-id', d_tree.get_file_revision('c'))