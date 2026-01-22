from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test___len__empty(self):
    chkmap = self._get_map({})
    self.assertEqual(0, len(chkmap))