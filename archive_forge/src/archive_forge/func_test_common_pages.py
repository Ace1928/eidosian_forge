from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_common_pages(self):
    basis = self.get_map_key({(b'a',): b'content', (b'b',): b'content', (b'c',): b'content'})
    target = self.get_map_key({(b'a',): b'content', (b'b',): b'other content', (b'c',): b'content'})
    target_map = CHKMap(self.get_chk_bytes(), target)
    self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('a',) 'content'\n  'b' LeafNode\n      ('b',) 'other content'\n  'c' LeafNode\n      ('c',) 'content'\n", target_map._dump_tree())
    b_key = target_map._root_node._items[b'b'].key()
    self.assertIterInteresting([target, b_key], [((b'b',), b'other content')], [target], [basis])