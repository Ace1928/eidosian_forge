from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_one_deep_map_plain(self):
    c_map = self.make_one_deep_map()
    self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('abb',) 'initial abb content'\n  'c' LeafNode\n      ('ccc',) 'initial ccc content'\n  'd' LeafNode\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())