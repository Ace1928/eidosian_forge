from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_current_size_items(self):
    node = LeafNode()
    base_size = node._current_size()
    node.map(None, (b'foo bar',), b'baz')
    self.assertEqual(base_size + 14, node._current_size())