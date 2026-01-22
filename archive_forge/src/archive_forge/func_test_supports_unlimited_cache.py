import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_supports_unlimited_cache(self):
    builder = btree_index.BTreeBuilder(reference_lists=0, key_elements=1)
    nodes = self.make_nodes(500, 1, 0)
    for node in nodes:
        builder.add_node(*node)
    stream = builder.finish()
    trans = self.get_transport()
    size = trans.put_file('index', stream)
    index = btree_index.BTreeGraphIndex(trans, 'index', size)
    self.assertEqual(500, index.key_count())
    self.assertEqual(2, len(index._row_lengths))
    self.assertTrue(index._row_lengths[-1] >= 2)
    self.assertIsInstance(index._leaf_node_cache, lru_cache.LRUCache)
    self.assertEqual(btree_index._NODE_CACHE_SIZE, index._leaf_node_cache._max_cache)
    self.assertIsInstance(index._internal_node_cache, fifo_cache.FIFOCache)
    self.assertEqual(100, index._internal_node_cache._max_cache)
    index = btree_index.BTreeGraphIndex(trans, 'index', size, unlimited_cache=False)
    self.assertIsInstance(index._leaf_node_cache, lru_cache.LRUCache)
    self.assertEqual(btree_index._NODE_CACHE_SIZE, index._leaf_node_cache._max_cache)
    self.assertIsInstance(index._internal_node_cache, fifo_cache.FIFOCache)
    self.assertEqual(100, index._internal_node_cache._max_cache)
    index = btree_index.BTreeGraphIndex(trans, 'index', size, unlimited_cache=True)
    self.assertIsInstance(index._leaf_node_cache, dict)
    self.assertIs(type(index._internal_node_cache), dict)
    entries = set(index.iter_entries([n[0] for n in nodes]))
    self.assertEqual(500, len(entries))