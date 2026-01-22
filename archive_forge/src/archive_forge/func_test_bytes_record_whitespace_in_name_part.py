from io import BytesIO
from ... import tests
from .. import pack
def test_bytes_record_whitespace_in_name_part(self):
    serialiser = pack.ContainerSerialiser()
    self.assertRaises(pack.InvalidRecordError, serialiser.bytes_record, b'bytes', [(b'bad name',)])