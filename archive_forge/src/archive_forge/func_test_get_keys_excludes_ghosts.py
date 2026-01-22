from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
def test_get_keys_excludes_ghosts(self):
    builder = self.make_branch_builder('b')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', ''))], revision_id=b'rev-1')
    builder.build_snapshot([b'rev-1', b'ghost'], [], revision_id=b'rev-2')
    builder.finish_series()
    repo = builder.get_branch().repository
    repo.lock_read()
    self.addCleanup(repo.unlock)
    result = vf_search.PendingAncestryResult([b'rev-2'], repo)
    self.assertEqual(sorted([b'rev-1', b'rev-2']), sorted(result.get_keys()))