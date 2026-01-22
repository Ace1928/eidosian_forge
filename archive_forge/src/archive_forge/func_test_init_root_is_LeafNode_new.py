from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_init_root_is_LeafNode_new(self):
    chk_bytes = self.get_chk_bytes()
    chkmap = CHKMap(chk_bytes, None)
    self.assertIsInstance(chkmap._root_node, LeafNode)
    self.assertEqual({}, self.to_dict(chkmap))
    self.assertEqual(0, len(chkmap))