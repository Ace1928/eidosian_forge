from io import BytesIO
from ... import tests
from .. import pack
def test_add_bytes_record_invalid_name(self):
    """Adding a Bytes record with a name with whitespace in it raises
        InvalidRecordError.
        """
    self.writer.begin()
    self.assertRaises(pack.InvalidRecordError, self.writer.add_bytes_record, [b'abc'], len(b'abc'), names=[(b'bad name',)])