from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_breadth_first_get_result_starting_a_ghost_ghost_is_excluded(self):
    graph = self.make_graph({b'head': [b'child'], b'child': [NULL_REVISION], NULL_REVISION: []})
    search = graph._make_breadth_first_searcher([b'head'])
    expected = [({b'head', b'ghost'}, ({b'head', b'ghost'}, {b'child', b'ghost'}, 1), [b'head'], [b'ghost'], None), ({b'head', b'child', b'ghost'}, ({b'head', b'ghost'}, {NULL_REVISION, b'ghost'}, 2), [b'head', b'child'], None, None)]
    self.assertSeenAndResult(expected, search, search.__next__)
    search = graph._make_breadth_first_searcher([b'head'])
    self.assertSeenAndResult(expected, search, search.next_with_ghosts)