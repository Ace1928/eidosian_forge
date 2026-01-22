import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test__sizeof__(self):
    leaf0 = self.module._parse_into_chk(b'type=leaf\n', 1, 0)
    leaf1 = self.module._parse_into_chk(_one_key_content, 1, 0)
    leafN = self.module._parse_into_chk(_multi_key_content, 1, 0)
    sizeof_1 = leaf1.__sizeof__() - leaf0.__sizeof__()
    self.assertTrue(sizeof_1 > 0)
    sizeof_N = leafN.__sizeof__() - leaf0.__sizeof__()
    self.assertEqual(sizeof_1 * len(leafN), sizeof_N)