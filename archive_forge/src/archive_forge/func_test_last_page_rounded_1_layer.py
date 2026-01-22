import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_last_page_rounded_1_layer(self):
    builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
    nodes = self.make_nodes(10, 1, 0)
    for node in nodes:
        builder.add_node(*node)
    temp_file = builder.finish()
    content = temp_file.read()
    del temp_file
    self.assertEqualApproxCompressed(155, len(content))
    self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=10\nrow_lengths=1\n', content[:74])
    leaf2 = content[74:]
    leaf2_bytes = zlib.decompress(leaf2)
    node = btree_index._LeafNode(leaf2_bytes, 1, 0)
    self.assertEqual(10, len(node))
    sorted_node_keys = sorted((node[0] for node in nodes))
    self.assertEqual(sorted_node_keys, node.all_keys())