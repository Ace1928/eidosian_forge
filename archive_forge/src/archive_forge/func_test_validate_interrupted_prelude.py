from io import BytesIO
from ... import tests
from .. import pack
def test_validate_interrupted_prelude(self):
    """EOF during reading a record's prelude causes validate to fail."""
    reader = self.get_reader_for(b'')
    self.assertRaises(pack.UnexpectedEndOfContainerError, reader.validate)