from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_get_cached_parent_map(self):
    self.assertEqual({}, self.caching_pp.get_cached_parent_map([b'a']))
    self.assertEqual([], self.inst_pp.calls)
    self.assertEqual({b'a': (b'b',)}, self.caching_pp.get_parent_map([b'a']))
    self.assertEqual([b'a'], self.inst_pp.calls)
    self.assertEqual({b'a': (b'b',)}, self.caching_pp.get_cached_parent_map([b'a']))