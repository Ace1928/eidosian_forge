from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_breadth_first_search_get_result_after_StopIteration(self):
    graph = self.make_graph({b'head': [NULL_REVISION], NULL_REVISION: []})
    search = graph._make_breadth_first_searcher([b'head'])
    expected = [({b'head'}, ({b'head'}, {NULL_REVISION}, 1), [b'head'], None, None), ({b'head', b'ghost', NULL_REVISION}, ({b'head', b'ghost'}, {b'ghost'}, 2), [b'head', NULL_REVISION], [b'ghost'], None)]
    self.assertSeenAndResult(expected, search, search.__next__)
    self.assertRaises(StopIteration, next, search)
    self.assertEqual({b'head', b'ghost', NULL_REVISION}, search.seen)
    state = search.get_state()
    self.assertEqual(({b'ghost', b'head'}, {b'ghost'}, {b'head', NULL_REVISION}), state)
    search = graph._make_breadth_first_searcher([b'head'])
    self.assertSeenAndResult(expected, search, search.next_with_ghosts)
    self.assertRaises(StopIteration, next, search)
    self.assertEqual({b'head', b'ghost', NULL_REVISION}, search.seen)
    state = search.get_state()
    self.assertEqual(({b'ghost', b'head'}, {b'ghost'}, {b'head', NULL_REVISION}), state)