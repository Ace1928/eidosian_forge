import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_from_invalid_hex(self):
    self.assertFailUnhexlify(b'123456789012345678901234567890123456789X')
    self.assertFailUnhexlify(b'12345678901234567890123456789012345678X9')