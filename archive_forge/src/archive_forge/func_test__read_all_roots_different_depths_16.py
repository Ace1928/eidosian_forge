from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__read_all_roots_different_depths_16(self):
    c_map = self.make_two_deep_map(chk_map._search_key_16)
    c_map._dump_tree()
    key1 = c_map.key()
    key1_2 = c_map._root_node._items[b'2'].key()
    key1_4 = c_map._root_node._items[b'4'].key()
    key1_C = c_map._root_node._items[b'C'].key()
    key1_F = c_map._root_node._items[b'F'].key()
    c_map2 = self.make_one_deep_two_prefix_map(chk_map._search_key_16)
    c_map2._dump_tree()
    key2 = c_map2.key()
    key2_F0 = c_map2._root_node._items[b'F0'].key()
    key2_F3 = c_map2._root_node._items[b'F3'].key()
    key2_F4 = c_map2._root_node._items[b'F4'].key()
    key2_FD = c_map2._root_node._items[b'FD'].key()
    diff = self.get_difference([key2], [key1], chk_map._search_key_16)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([key2], root_results)
    self.assertEqual([key1_F], diff._old_queue)
    self.assertEqual(sorted([key2_F0, key2_F3, key2_F4, key2_FD]), sorted(diff._new_queue))
    self.assertEqual([], diff._new_item_queue)
    diff = self.get_difference([key1], [key2], chk_map._search_key_16)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([key1], root_results)
    self.assertEqual(sorted([key2_F0, key2_F3, key2_F4, key2_FD]), sorted(diff._old_queue))
    self.assertEqual(sorted([key1_2, key1_4, key1_C, key1_F]), sorted(diff._new_queue))
    self.assertEqual([], diff._new_item_queue)