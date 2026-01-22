import hashlib
import zlib
from .. import estimate_compressed_size, tests
def test_adding_more_content(self):
    ze = estimate_compressed_size.ZLibEstimator(64000)
    raw_data = self.get_slightly_random_content(150000)
    block_size = 1000
    for start in range(0, len(raw_data), block_size):
        ze.add_content(raw_data[start:start + block_size])
        if ze.full():
            break
    self.assertTrue(110000 <= start <= 114000, 'Unexpected amount of raw data added: %d bytes' % (start,))
    raw_comp = zlib.compress(raw_data[:start])
    self.assertTrue(63000 < len(raw_comp) < 65000, 'Unexpected compressed size: %d bytes' % (len(raw_comp),))