from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_one_deep_two_prefix_map_16(self):
    c_map = self.make_one_deep_two_prefix_map(search_key_func=chk_map._search_key_16)
    self.assertEqualDiff("'' InternalNode\n  'F0' LeafNode\n      ('aaa',) 'initial aaa content'\n  'F3' LeafNode\n      ('adl',) 'initial adl content'\n  'F4' LeafNode\n      ('adh',) 'initial adh content'\n  'FD' LeafNode\n      ('add',) 'initial add content'\n", c_map._dump_tree())