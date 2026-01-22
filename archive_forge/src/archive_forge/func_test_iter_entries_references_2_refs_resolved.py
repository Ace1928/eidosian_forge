import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_iter_entries_references_2_refs_resolved(self):
    builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
    nodes = self.make_nodes(160, 2, 2)
    for node in nodes:
        builder.add_node(*node)
    t = transport.get_transport_from_url('trace+' + self.get_url(''))
    size = t.put_file('index', builder.finish())
    del builder
    index = btree_index.BTreeGraphIndex(t, 'index', size)
    del t._activity[:]
    self.assertEqual([], t._activity)
    found_nodes = list(index.iter_entries([nodes[30][0]]))
    bare_nodes = []
    for node in found_nodes:
        self.assertTrue(node[0] is index)
        bare_nodes.append(node[1:])
    self.assertEqual(1, len(found_nodes))
    self.assertEqual(nodes[30], bare_nodes[0])
    self.assertEqual([('readv', 'index', [(0, 4096)], False, None), ('readv', 'index', [(8192, 4096)], False, None)], t._activity)