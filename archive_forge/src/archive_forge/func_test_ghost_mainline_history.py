from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_ghost_mainline_history(self):
    builder = BranchBuilder(self.get_transport().clone('foo'))
    builder.start_series()
    try:
        builder.build_snapshot([b'ghost'], [('add', ('', b'ROOT_ID', 'directory', ''))], allow_leftmost_as_ghost=True, revision_id=b'tip')
    finally:
        builder.finish_series()
    b = builder.get_branch()
    b.lock_read()
    self.addCleanup(b.unlock)
    self.assertEqual((b'ghost',), b.repository.get_graph().get_parent_map([b'tip'])[b'tip'])