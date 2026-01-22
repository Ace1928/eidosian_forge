import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_2_leaves_2_2(self):
    builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
    nodes = self.make_nodes(100, 2, 2)
    for node in nodes:
        builder.add_node(*node)
    temp_file = builder.finish()
    content = temp_file.read()
    del temp_file
    self.assertEqualApproxCompressed(12643, len(content))
    self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=2\nkey_elements=2\nlen=200\nrow_lengths=1,3\n', content[:77])
    root = content[77:4096]
    leaf1 = content[4096:8192]
    leaf2 = content[8192:12288]
    leaf3 = content[12288:]
    root_bytes = zlib.decompress(root)
    expected_root = b'type=internal\noffset=0\n' + b'0' * 40 + b'\x00' + b'91' * 40 + b'\n' + b'1' * 40 + b'\x00' + b'81' * 40 + b'\n'
    self.assertEqual(expected_root, root_bytes)