from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
def test_get_keys_excludes_null(self):

    class StubGraph:

        def iter_ancestry(self, keys):
            return [(NULL_REVISION, ()), (b'foo', (NULL_REVISION,))]
    result = vf_search.PendingAncestryResult([b'rev-3'], None)
    result_keys = result._get_keys(StubGraph())
    self.assertEqual({b'foo'}, set(result_keys))