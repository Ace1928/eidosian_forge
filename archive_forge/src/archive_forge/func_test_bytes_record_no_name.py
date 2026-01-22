from io import BytesIO
from ... import tests
from .. import pack
def test_bytes_record_no_name(self):
    serialiser = pack.ContainerSerialiser()
    record = serialiser.bytes_record(b'bytes', [])
    self.assertEqual(b'B5\n\nbytes', record)