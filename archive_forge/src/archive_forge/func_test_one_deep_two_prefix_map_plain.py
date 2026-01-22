from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_one_deep_two_prefix_map_plain(self):
    c_map = self.make_one_deep_two_prefix_map()
    self.assertEqualDiff("'' InternalNode\n  'aa' LeafNode\n      ('aaa',) 'initial aaa content'\n  'ad' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n", c_map._dump_tree())