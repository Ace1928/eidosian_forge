from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_map_exceeding_max_size_only_entry_new(self):
    node = LeafNode()
    node.set_maximum_size(10)
    result = node.map(None, (b'foo bar',), b'baz quux')
    self.assertEqual((b'foo bar', [(b'', node)]), result)
    self.assertTrue(10 < node._current_size())