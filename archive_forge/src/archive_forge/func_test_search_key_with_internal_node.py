from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_search_key_with_internal_node(self):
    chk_bytes = self.get_chk_bytes()
    chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=_test_search_key)
    chkmap._root_node.set_maximum_size(10)
    chkmap.map((b'1',), b'foo')
    chkmap.map((b'2',), b'bar')
    chkmap.map((b'3',), b'baz')
    self.assertEqualDiff("'' InternalNode\n  'test:1' LeafNode\n      ('1',) 'foo'\n  'test:2' LeafNode\n      ('2',) 'bar'\n  'test:3' LeafNode\n      ('3',) 'baz'\n", chkmap._dump_tree())
    root_key = chkmap._save()
    chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=_test_search_key)
    self.assertEqualDiff("'' InternalNode\n  'test:1' LeafNode\n      ('1',) 'foo'\n  'test:2' LeafNode\n      ('2',) 'bar'\n  'test:3' LeafNode\n      ('3',) 'baz'\n", chkmap._dump_tree())