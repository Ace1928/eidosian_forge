from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__iter_nodes_mixed_key_width(self):
    node = self.make_fo_fa_node()
    key_filter = [(b'foo', b'bar'), (b'foo',), (b'fo',), (b'b',)]
    nodes = list(node._iter_nodes(None, key_filter=key_filter))
    self.assertEqual(1, len(nodes))
    matches = key_filter[:]
    matches.remove((b'b',))
    self.assertEqual(sorted(matches), sorted(nodes[0][1]))