from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_set_parent_to_null(self):
    builder = self.build_a_rev()
    builder.start_series()
    self.addCleanup(builder.finish_series)
    builder.build_snapshot([], [('add', ('', None, 'directory', None))], revision_id=b'B-id')
    repo = builder.get_branch().repository
    self.assertEqual({b'A-id': (_mod_revision.NULL_REVISION,), b'B-id': (_mod_revision.NULL_REVISION,)}, repo.get_parent_map([b'A-id', b'B-id']))