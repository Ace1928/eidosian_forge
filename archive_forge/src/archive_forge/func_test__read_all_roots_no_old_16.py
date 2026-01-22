from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__read_all_roots_no_old_16(self):
    c_map = self.make_two_deep_map(chk_map._search_key_16)
    key1 = c_map.key()
    diff = self.get_difference([key1], [], chk_map._search_key_16)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([], root_results)
    self.assertEqual([], diff._old_queue)
    self.assertEqual([key1], diff._new_queue)
    self.assertEqual([], diff._new_item_queue)
    c_map2 = self.make_one_deep_map(chk_map._search_key_16)
    key2 = c_map2.key()
    diff = self.get_difference([key1, key2], [], chk_map._search_key_16)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([], root_results)
    self.assertEqual([], diff._old_queue)
    self.assertEqual(sorted([key1, key2]), sorted(diff._new_queue))
    self.assertEqual([], diff._new_item_queue)