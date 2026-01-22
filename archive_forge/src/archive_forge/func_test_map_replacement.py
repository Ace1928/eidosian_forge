from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_map_replacement(self):
    node = LeafNode()
    node.map(None, (b'foo bar',), b'baz quux')
    result = node.map(None, (b'foo bar',), b'bango')
    self.assertEqual((b'foo bar', [(b'', node)]), result)
    self.assertEqual({(b'foo bar',): b'bango'}, self.to_dict(node, None))
    self.assertEqual(1, len(node))