from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_get_parent_map_repeated(self):
    """Asking for the same parent 2x will only forward 1 request."""
    self.assertEqual({b'a': (b'b',)}, self.caching_pp.get_parent_map([b'b', b'a', b'b']))
    self.assertEqual([b'a', b'b'], sorted(self.inst_pp.calls))