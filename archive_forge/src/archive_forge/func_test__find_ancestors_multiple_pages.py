import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test__find_ancestors_multiple_pages(self):
    start_time = 1249671539
    email = 'joebob@example.com'
    nodes = []
    ref_lists = ((),)
    rev_keys = []
    for i in range(400):
        rev_id = '{}-{}-{}'.format(email, osutils.compact_date(start_time + i), osutils.rand_chars(16)).encode('ascii')
        rev_key = (rev_id,)
        nodes.append((rev_key, b'value', ref_lists))
        ref_lists = ((rev_key,),)
        rev_keys.append(rev_key)
    index = self.make_index(ref_lists=1, key_elements=1, nodes=nodes)
    self.assertEqual(400, index.key_count())
    self.assertEqual(3, len(index._row_offsets))
    nodes = dict(index._read_nodes([1, 2]))
    l1 = nodes[1]
    l2 = nodes[2]
    min_l2_key = l2.min_key
    max_l1_key = l1.max_key
    self.assertTrue(max_l1_key < min_l2_key)
    parents_min_l2_key = l2[min_l2_key][1][0]
    self.assertEqual((l1.max_key,), parents_min_l2_key)
    key_idx = rev_keys.index(min_l2_key)
    next_key = rev_keys[key_idx + 1]
    parent_map = {}
    missing_keys = set()
    search_keys = index._find_ancestors([next_key], 0, parent_map, missing_keys)
    self.assertEqual([min_l2_key, next_key], sorted(parent_map))
    self.assertEqual(set(), missing_keys)
    self.assertEqual({max_l1_key}, search_keys)
    parent_map = {}
    search_keys = index._find_ancestors([max_l1_key], 0, parent_map, missing_keys)
    self.assertEqual(l1.all_keys(), sorted(parent_map))
    self.assertEqual(set(), missing_keys)
    self.assertEqual(set(), search_keys)