from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unique_serialised_prefix_one_item_new(self):
    node = LeafNode()
    node.map(None, (b'foo bar', b'baz'), b'baz quux')
    self.assertEqual(b'foo bar\x00baz', node._compute_search_prefix())