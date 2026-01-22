from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_unversion_root_add_new_root(self):
    builder = BranchBuilder(self.get_transport().clone('foo'))
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'TREE_ROOT', 'directory', ''))], revision_id=b'rev-1')
    builder.build_snapshot(None, [('unversion', ''), ('add', ('', b'my-root', 'directory', ''))], revision_id=b'rev-2')
    builder.finish_series()
    rev_tree = builder.get_branch().repository.revision_tree(b'rev-2')
    self.assertTreeShape([('', b'my-root', 'directory')], rev_tree)