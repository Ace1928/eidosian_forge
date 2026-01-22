from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_apply_ab_empty(self):
    chk_bytes = self.get_chk_bytes()
    root_key = CHKMap.from_dict(chk_bytes, {(b'a',): b'b'})
    chkmap = CHKMap(chk_bytes, root_key)
    new_root = chkmap.apply_delta([((b'a',), None, None)])
    expected_root_key = self.assertHasEmptyMap(chk_bytes)
    self.assertEqual(expected_root_key, new_root)
    self.assertEqual(new_root, chkmap._root_node._key)