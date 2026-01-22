from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_two_deep_map_plain(self):
    c_map = self.make_two_deep_map()
    self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aa' LeafNode\n      ('aaa',) 'initial aaa content'\n    'ab' LeafNode\n      ('abb',) 'initial abb content'\n    'ac' LeafNode\n      ('acc',) 'initial acc content'\n      ('ace',) 'initial ace content'\n    'ad' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n  'c' LeafNode\n      ('ccc',) 'initial ccc content'\n  'd' LeafNode\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())