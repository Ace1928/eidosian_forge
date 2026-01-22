import gzip
from io import BytesIO
from .. import tests, tuned_gzip
def test_enormous_chunks(self):
    self.assertToGzip([b'a large string\n' * 1024 * 256])
    self.assertToGzip([b'a large string\n'] * 1024 * 256)