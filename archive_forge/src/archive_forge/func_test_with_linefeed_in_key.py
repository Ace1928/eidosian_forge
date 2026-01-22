from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_with_linefeed_in_key(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(10)
    chkmap.map((b'a\ra',), b'val1')
    chkmap.map((b'a\rb',), b'val2')
    chkmap.map((b'ac',), b'val3')
    self.assertCanonicalForm(chkmap)
    self.assertEqualDiff("'' InternalNode\n  'a\\r' InternalNode\n    'a\\ra' LeafNode\n      ('a\\ra',) 'val1'\n    'a\\rb' LeafNode\n      ('a\\rb',) 'val2'\n  'ac' LeafNode\n      ('ac',) 'val3'\n", chkmap._dump_tree())
    root_key = chkmap._save()
    chkmap = CHKMap(store, root_key)
    self.assertEqualDiff("'' InternalNode\n  'a\\r' InternalNode\n    'a\\ra' LeafNode\n      ('a\\ra',) 'val1'\n    'a\\rb' LeafNode\n      ('a\\rb',) 'val2'\n  'ac' LeafNode\n      ('ac',) 'val3'\n", chkmap._dump_tree())