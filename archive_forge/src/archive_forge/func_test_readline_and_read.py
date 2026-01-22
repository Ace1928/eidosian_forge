from io import BytesIO
from ... import tests
from .. import pack
def test_readline_and_read(self):
    """Test exercising one byte reads, readline, and then read again."""
    transport = self.get_transport()
    transport.put_bytes('sample', b'0\n2\n4\n')
    f = pack.ReadVFile(transport.readv('sample', [(0, 6)]))
    results = []
    results.append(f.read(1))
    results.append(f.readline())
    results.append(f.read(4))
    self.assertEqual([b'0', b'\n', b'2\n4\n'], results)