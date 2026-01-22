from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_map_splits_with_longer_key(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(10)
    chkmap.map((b'aaa',), b'v')
    chkmap.map((b'aaaa',), b'v')
    self.assertCanonicalForm(chkmap)
    self.assertIsInstance(chkmap._root_node, InternalNode)