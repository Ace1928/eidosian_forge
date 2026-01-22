from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unmap_double_deep_non_empty_leaf(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(40)
    chkmap.map((b'aaa',), b'v')
    chkmap.map((b'aab',), b'long value')
    chkmap.map((b'aabb',), b'v')
    chkmap.map((b'abc',), b'v')
    self.assertEqualDiff("'' InternalNode\n  'aa' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'v'\n    'aab' LeafNode\n      ('aab',) 'long value'\n      ('aabb',) 'v'\n  'ab' LeafNode\n      ('abc',) 'v'\n", chkmap._dump_tree())
    chkmap.unmap((b'aab',))
    self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aabb',) 'v'\n      ('abc',) 'v'\n", chkmap._dump_tree())