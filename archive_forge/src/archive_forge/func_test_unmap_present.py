from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unmap_present(self):
    node = LeafNode()
    node.map(None, (b'foo bar',), b'baz quux')
    result = node.unmap(None, (b'foo bar',))
    self.assertEqual(node, result)
    self.assertEqual({}, self.to_dict(node, None))
    self.assertEqual(0, len(node))