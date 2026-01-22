from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unmap_double_deep(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(40)
    chkmap.map((b'aaa',), b'v')
    chkmap.map((b'aaab',), b'v')
    chkmap.map((b'aab',), b'very long value')
    chkmap.map((b'abc',), b'v')
    self.assertEqualDiff("'' InternalNode\n  'aa' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'v'\n      ('aaab',) 'v'\n    'aab' LeafNode\n      ('aab',) 'very long value'\n  'ab' LeafNode\n      ('abc',) 'v'\n", chkmap._dump_tree())
    chkmap.unmap((b'aab',))
    self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aaab',) 'v'\n      ('abc',) 'v'\n", chkmap._dump_tree())