from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__search_prefix_filter_with_hash(self):
    search_key_func = chk_map.search_key_registry.get(b'hash-16-way')
    node = InternalNode(search_key_func=search_key_func)
    node._key_width = 2
    node._node_width = 4
    self.assertEqual(b'E8B7BE43\x0071BEEFF9', search_key_func(StaticTuple(b'a', b'b')))
    self.assertEqual(b'E8B7', node._search_prefix_filter(StaticTuple(b'a', b'b')))
    self.assertEqual(b'E8B7', node._search_prefix_filter(StaticTuple(b'a')))