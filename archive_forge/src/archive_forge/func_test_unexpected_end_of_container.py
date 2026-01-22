from io import BytesIO
from ... import tests
from .. import pack
def test_unexpected_end_of_container(self):
    """Test the formatting of UnexpectedEndOfContainerError."""
    e = pack.UnexpectedEndOfContainerError()
    self.assertEqual('Unexpected end of container stream', str(e))