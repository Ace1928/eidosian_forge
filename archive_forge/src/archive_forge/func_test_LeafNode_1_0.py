import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_LeafNode_1_0(self):
    node_bytes = b'type=leaf\n0000000000000000000000000000000000000000\x00\x00value:0\n1111111111111111111111111111111111111111\x00\x00value:1\n2222222222222222222222222222222222222222\x00\x00value:2\n3333333333333333333333333333333333333333\x00\x00value:3\n4444444444444444444444444444444444444444\x00\x00value:4\n'
    node = btree_index._LeafNode(node_bytes, 1, 0)
    self.assertEqual({(b'0000000000000000000000000000000000000000',): (b'value:0', ()), (b'1111111111111111111111111111111111111111',): (b'value:1', ()), (b'2222222222222222222222222222222222222222',): (b'value:2', ()), (b'3333333333333333333333333333333333333333',): (b'value:3', ()), (b'4444444444444444444444444444444444444444',): (b'value:4', ())}, dict(node.all_items()))