from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_start_finish_series(self):
    builder = BranchBuilder(self.get_transport().clone('foo'))
    builder.start_series()
    try:
        self.assertIsNot(None, builder._tree)
        self.assertEqual('w', builder._tree._lock_mode)
        self.assertTrue(builder._branch.is_locked())
    finally:
        builder.finish_series()
    self.assertIs(None, builder._tree)
    self.assertFalse(builder._branch.is_locked())