import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_all_common_prefix(self):
    leaf = self.module._parse_into_chk(_common_32_bits, 1, 0)
    self.assertEqual(0, leaf.common_shift)
    lst = [120] * 8
    offsets = leaf._get_offsets()
    self.assertEqual([bisect.bisect_left(lst, x) for x in range(0, 257)], offsets)
    for val in lst:
        self.assertEqual(lst.index(val), offsets[val])
    for idx, key in enumerate(leaf.all_keys()):
        self.assertEqual(b'%d' % idx, leaf[key][0].split()[0])