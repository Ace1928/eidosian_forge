from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_multiple_maps_overlapping_common_new(self):
    basis = self.get_map_key({(b'aaa',): b'left', (b'abb',): b'right', (b'ccc',): b'common'})
    left = self.get_map_key({(b'aaa',): b'left', (b'abb',): b'right', (b'ccc',): b'common', (b'ddd',): b'change'})
    right = self.get_map_key({(b'abb',): b'right'})
    basis_map = CHKMap(self.get_chk_bytes(), basis)
    self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aa' LeafNode\n      ('aaa',) 'left'\n    'ab' LeafNode\n      ('abb',) 'right'\n  'c' LeafNode\n      ('ccc',) 'common'\n", basis_map._dump_tree())
    left_map = CHKMap(self.get_chk_bytes(), left)
    self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aa' LeafNode\n      ('aaa',) 'left'\n    'ab' LeafNode\n      ('abb',) 'right'\n  'c' LeafNode\n      ('ccc',) 'common'\n  'd' LeafNode\n      ('ddd',) 'change'\n", left_map._dump_tree())
    l_d_key = left_map._root_node._items[b'd'].key()
    right_map = CHKMap(self.get_chk_bytes(), right)
    self.assertEqualDiff("'' LeafNode\n      ('abb',) 'right'\n", right_map._dump_tree())
    self.assertIterInteresting([right, left, l_d_key], [((b'ddd',), b'change')], [left, right], [basis])