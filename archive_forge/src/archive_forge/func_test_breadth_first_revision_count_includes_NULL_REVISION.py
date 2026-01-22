from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_breadth_first_revision_count_includes_NULL_REVISION(self):
    graph = self.make_graph({b'head': [NULL_REVISION], NULL_REVISION: []})
    search = graph._make_breadth_first_searcher([b'head'])
    expected = [({b'head'}, ({b'head'}, {NULL_REVISION}, 1), [b'head'], None, None), ({b'head', NULL_REVISION}, ({b'head'}, set(), 2), [b'head', NULL_REVISION], None, None)]
    self.assertSeenAndResult(expected, search, search.__next__)
    search = graph._make_breadth_first_searcher([b'head'])
    self.assertSeenAndResult(expected, search, search.next_with_ghosts)