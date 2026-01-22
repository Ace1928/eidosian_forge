from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_default_chk_map_uses_flat_search_key(self):
    chkmap = chk_map.CHKMap(self.get_chk_bytes(), None)
    self.assertEqual(b'1', chkmap._search_key_func((b'1',)))
    self.assertEqual(b'1\x002', chkmap._search_key_func((b'1', b'2')))
    self.assertEqual(b'1\x002\x003', chkmap._search_key_func((b'1', b'2', b'3')))