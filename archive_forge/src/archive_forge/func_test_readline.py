from io import BytesIO
from ... import tests
from .. import pack
def test_readline(self):
    """Test using readline() as ContainerReader does.

        This is always within a readv hunk, never across it.
        """
    transport = self.get_transport()
    transport.put_bytes('sample', b'0\n2\n4\n')
    f = pack.ReadVFile(transport.readv('sample', [(0, 2), (2, 4)]))
    results = []
    results.append(f.readline())
    results.append(f.readline())
    results.append(f.readline())
    self.assertEqual([b'0\n', b'2\n', b'4\n'], results)