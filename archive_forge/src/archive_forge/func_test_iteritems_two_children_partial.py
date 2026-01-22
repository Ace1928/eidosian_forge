from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iteritems_two_children_partial(self):
    node = InternalNode()
    leaf1 = LeafNode()
    leaf1.map(None, (b'foo bar',), b'quux')
    leaf2 = LeafNode()
    leaf2.map(None, (b'strange',), b'beast')
    node.add_node(b'f', leaf1)
    node._items[b'f'] = None
    node.add_node(b's', leaf2)
    self.assertEqual([((b'strange',), b'beast')], sorted(node.iteritems(None, [(b'strange',), (b'weird',)])))