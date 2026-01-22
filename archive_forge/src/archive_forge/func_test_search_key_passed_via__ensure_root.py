from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_search_key_passed_via__ensure_root(self):
    chk_bytes = self.get_chk_bytes()
    chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=_test_search_key)
    root_key = chkmap._save()
    chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=_test_search_key)
    chkmap._ensure_root()
    self.assertEqual(b'test:1\x002\x003', chkmap._root_node._search_key((b'1', b'2', b'3')))