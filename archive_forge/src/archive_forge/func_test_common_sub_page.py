from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_common_sub_page(self):
    basis = self.get_map_key({(b'aaa',): b'common', (b'c',): b'common'})
    target = self.get_map_key({(b'aaa',): b'common', (b'aab',): b'new', (b'c',): b'common'})
    target_map = CHKMap(self.get_chk_bytes(), target)
    self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'common'\n    'aab' LeafNode\n      ('aab',) 'new'\n  'c' LeafNode\n      ('c',) 'common'\n", target_map._dump_tree())
    a_key = target_map._root_node._items[b'a'].key()
    aab_key = target_map._root_node._items[b'a']._items[b'aab'].key()
    self.assertIterInteresting([target, a_key, aab_key], [((b'aab',), b'new')], [target], [basis])