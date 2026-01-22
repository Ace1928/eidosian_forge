from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unmap_k1_from_k1_k22_k23_gives_k22_k23_tree_new(self):
    chkmap = self._get_map({(b'k1',): b'foo', (b'k22',): b'bar', (b'k23',): b'quux'}, maximum_size=10)
    self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' InternalNode\n    'k22' LeafNode\n      ('k22',) 'bar'\n    'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())
    orig_root = chkmap._root_node
    chkmap = CHKMap(chkmap._store, orig_root)
    chkmap._ensure_root()
    node = chkmap._root_node
    k2_ptr = node._items[b'k2']
    result = node.unmap(chkmap._store, (b'k1',))
    self.assertEqual(k2_ptr, result)
    chkmap = CHKMap(chkmap._store, orig_root)
    chkmap.unmap((b'k1',))
    self.assertEqual(k2_ptr, chkmap._root_node)
    self.assertEqualDiff("'' InternalNode\n  'k22' LeafNode\n      ('k22',) 'bar'\n  'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())