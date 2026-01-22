from io import BytesIO
from ... import tests
from .. import pack
def test_validate_whitespace_in_name(self):
    """Names must have no whitespace."""
    reader = self.get_reader_for(b'0\nbad name\n\n')
    self.assertRaises(pack.InvalidRecordError, reader.validate)