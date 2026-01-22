from io import BytesIO
from ... import tests
from .. import pack
def test_add_bytes_record_no_name(self):
    """Add a bytes record with no name."""
    self.writer.begin()
    offset, length = self.writer.add_bytes_record([b'abc'], len(b'abc'), names=[])
    self.assertEqual((42, 7), (offset, length))
    self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nB3\n\nabc')