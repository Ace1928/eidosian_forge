import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_with_size_constructor(self):
    t = transport.get_transport_from_url('trace+' + self.get_url(''))
    index = btree_index.BTreeGraphIndex(t, 'index', 1)
    self.assertEqual([], t._activity)