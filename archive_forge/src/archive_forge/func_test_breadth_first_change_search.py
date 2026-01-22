from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_breadth_first_change_search(self):
    graph = self.make_graph({b'head': [b'present'], b'present': [b'stopped'], b'other': [b'other_2'], b'other_2': []})
    search = graph._make_breadth_first_searcher([b'head'])
    self.assertEqual(({b'head'}, set()), search.next_with_ghosts())
    self.assertEqual(({b'present'}, set()), search.next_with_ghosts())
    self.assertEqual({b'present'}, search.stop_searching_any([b'present']))
    self.assertEqual(({b'other'}, {b'other_ghost'}), search.start_searching([b'other', b'other_ghost']))
    self.assertEqual(({b'other_2'}, set()), search.next_with_ghosts())
    self.assertRaises(StopIteration, search.next_with_ghosts)
    search = graph._make_breadth_first_searcher([b'head'])
    self.assertEqual({b'head'}, next(search))
    self.assertEqual({b'present'}, next(search))
    self.assertEqual({b'present'}, search.stop_searching_any([b'present']))
    search.start_searching([b'other', b'other_ghost'])
    self.assertEqual({b'other_2'}, next(search))
    self.assertRaises(StopIteration, next, search)