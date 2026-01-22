import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_iter_all_only_root_no_size(self):
    self.make_index(nodes=[((b'key',), b'value', ())])
    t = transport.get_transport_from_url('trace+' + self.get_url(''))
    index = btree_index.BTreeGraphIndex(t, 'index', None)
    del t._activity[:]
    self.assertEqual([((b'key',), b'value')], [x[1:] for x in index.iter_all_entries()])
    self.assertEqual([('get', 'index')], t._activity)