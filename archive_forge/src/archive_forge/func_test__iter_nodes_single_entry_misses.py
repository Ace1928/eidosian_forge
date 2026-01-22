from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__iter_nodes_single_entry_misses(self):
    node = self.make_fo_fa_node()
    key_filter = [(b'bar',)]
    nodes = list(node._iter_nodes(None, key_filter=key_filter))
    self.assertEqual(0, len(nodes))