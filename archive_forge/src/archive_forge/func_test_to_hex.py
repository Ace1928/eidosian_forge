import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_to_hex(self):
    raw_bytes = bytes(range(256))
    for i in range(0, 240, 20):
        self.assertHexlify(raw_bytes[i:i + 20])
    self.assertHexlify(raw_bytes[240:] + raw_bytes[0:4])