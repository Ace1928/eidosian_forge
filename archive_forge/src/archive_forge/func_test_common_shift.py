import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_common_shift(self):
    leaf = self.module._parse_into_chk(_multi_key_content, 1, 0)
    self.assertEqual(19, leaf.common_shift)
    lst = [1, 13, 28, 180, 190, 193, 210, 239]
    offsets = leaf._get_offsets()
    self.assertEqual([bisect.bisect_left(lst, x) for x in range(0, 257)], offsets)
    for idx, val in enumerate(lst):
        self.assertEqual(idx, offsets[val])
    for idx, key in enumerate(leaf.all_keys()):
        self.assertEqual(b'%d' % idx, leaf[key][0].split()[0])