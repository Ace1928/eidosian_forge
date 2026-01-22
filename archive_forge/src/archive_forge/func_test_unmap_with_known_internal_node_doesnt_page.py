from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unmap_with_known_internal_node_doesnt_page(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(30)
    chkmap.map((b'aaa',), b'v')
    chkmap.map((b'aab',), b'v')
    chkmap.map((b'aac',), b'v')
    chkmap.map((b'abc',), b'v')
    chkmap.map((b'acd',), b'v')
    self.assertEqualDiff("'' InternalNode\n  'aa' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'v'\n    'aab' LeafNode\n      ('aab',) 'v'\n    'aac' LeafNode\n      ('aac',) 'v'\n  'ab' LeafNode\n      ('abc',) 'v'\n  'ac' LeafNode\n      ('acd',) 'v'\n", chkmap._dump_tree())
    chkmap = CHKMap(store, chkmap._save())
    chkmap.map((b'aad',), b'v')
    self.assertIsInstance(chkmap._root_node._items[b'aa'], InternalNode)
    self.assertIsInstance(chkmap._root_node._items[b'ab'], StaticTuple)
    self.assertIsInstance(chkmap._root_node._items[b'ac'], StaticTuple)
    chkmap.unmap((b'acd',))
    self.assertIsInstance(chkmap._root_node._items[b'aa'], InternalNode)
    self.assertIsInstance(chkmap._root_node._items[b'ab'], StaticTuple)