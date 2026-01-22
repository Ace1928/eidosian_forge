from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_current_size_width_changed(self):
    node = LeafNode()
    node._key_width = 10
    self.assertEqual(17, node._current_size())