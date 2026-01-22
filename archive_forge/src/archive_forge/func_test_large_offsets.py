import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_large_offsets(self):
    leaf = self.module._parse_into_chk(_large_offsets, 1, 0)
    self.assertEqual([b'12345678901 1234567890 0 1', b'2147483648 2147483647 0 1', b'4294967296 4294967295 4294967294 1'], [x[1][0] for x in leaf.all_items()])