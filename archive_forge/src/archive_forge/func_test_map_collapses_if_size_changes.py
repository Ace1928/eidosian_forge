from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_map_collapses_if_size_changes(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(35)
    chkmap.map((b'aaa',), b'v')
    chkmap.map((b'aab',), b'very long value that splits')
    self.assertEqualDiff("'' InternalNode\n  'aaa' LeafNode\n      ('aaa',) 'v'\n  'aab' LeafNode\n      ('aab',) 'very long value that splits'\n", chkmap._dump_tree())
    self.assertCanonicalForm(chkmap)
    chkmap.map((b'aab',), b'v')
    self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aab',) 'v'\n", chkmap._dump_tree())
    self.assertCanonicalForm(chkmap)