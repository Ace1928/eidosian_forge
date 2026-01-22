from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__iter_nodes_with_multiple_matches(self):
    node = InternalNode(b'')
    child = LeafNode()
    child.set_maximum_size(100)
    child.map(None, (b'foo',), b'val')
    child.map(None, (b'fob',), b'val')
    node.add_node(b'f', child)
    child = LeafNode()
    child.set_maximum_size(100)
    child.map(None, (b'bar',), b'val')
    child.map(None, (b'baz',), b'val')
    node.add_node(b'b', child)
    key_filter = ((b'foo',), (b'fob',), (b'bar',), (b'baz',), (b'ram',))
    for child, node_key_filter in node._iter_nodes(None, key_filter=key_filter):
        self.assertEqual(2, len(node_key_filter))