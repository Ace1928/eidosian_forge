from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_one_deep_map_16(self):
    c_map = self.make_one_deep_map(search_key_func=chk_map._search_key_16)
    self.assertEqualDiff("'' InternalNode\n  '2' LeafNode\n      ('ccc',) 'initial ccc content'\n  '4' LeafNode\n      ('abb',) 'initial abb content'\n  'F' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())