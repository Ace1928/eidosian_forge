from io import BytesIO
from ... import tests
from .. import pack
def test_eof_after_length(self):
    """EOF after reading the length and before reading name(s)."""
    reader = self.get_reader_for(b'123\n')
    self.assertRaises(pack.UnexpectedEndOfContainerError, reader.read)