from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__iter_nodes_match_all(self):
    node = self.make_fo_fa_node()
    key_filter = [(b'foo', b'bar'), (b'foo',), (b'fo',), (b'f',)]
    nodes = list(node._iter_nodes(None, key_filter=key_filter))
    self.assertEqual(2, len(nodes))