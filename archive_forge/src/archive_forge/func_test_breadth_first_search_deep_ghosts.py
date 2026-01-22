from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_breadth_first_search_deep_ghosts(self):
    graph = self.make_graph({b'head': [b'present'], b'present': [b'child', b'ghost'], b'child': []})
    search = graph._make_breadth_first_searcher([b'head'])
    self.assertEqual(({b'head'}, set()), search.next_with_ghosts())
    self.assertEqual(({b'present'}, set()), search.next_with_ghosts())
    self.assertEqual(({b'child'}, {b'ghost'}), search.next_with_ghosts())
    self.assertRaises(StopIteration, search.next_with_ghosts)
    search = graph._make_breadth_first_searcher([b'head'])
    self.assertEqual({b'head'}, next(search))
    self.assertEqual({b'present'}, next(search))
    self.assertEqual({b'child', b'ghost'}, next(search))
    self.assertRaises(StopIteration, next, search)