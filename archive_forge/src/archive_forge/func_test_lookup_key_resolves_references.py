from ... import errors, tests, transport
from .. import index as _mod_index
def test_lookup_key_resolves_references(self):
    nodes = []
    for counter in range(99):
        nodes.append((self.make_key(counter), self.make_value(counter), ((self.make_key(counter + 20),),)))
    index = self.make_index(ref_lists=1, nodes=nodes)
    index_size = index._size
    index_center = index_size // 2
    result = index._lookup_keys_via_location([(index_center, (b'40',))])
    self.assertEqual([(0, 4027), (10198, 14028)], index._parsed_byte_map)
    self.assertEqual([((), self.make_key(17)), (self.make_key(44), self.make_key(5))], index._parsed_key_map)
    self.assertEqual([('readv', 'index', [(index_center, 800), (0, 200)], True, index_size)], index._transport._activity)
    del index._transport._activity[:]
    result = index._lookup_keys_via_location([(11000, self.make_key(45))])
    self.assertEqual([((11000, self.make_key(45)), (index, self.make_key(45), self.make_value(45), ((self.make_key(65),),)))], result)
    self.assertEqual([('readv', 'index', [(15093, 800)], True, index_size)], index._transport._activity)