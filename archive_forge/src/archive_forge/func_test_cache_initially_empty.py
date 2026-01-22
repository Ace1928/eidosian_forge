from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_cache_initially_empty(self):
    self.assertEqual({}, self.caching_pp._cache)