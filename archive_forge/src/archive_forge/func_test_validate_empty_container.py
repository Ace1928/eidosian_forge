from io import BytesIO
from ... import tests
from .. import pack
def test_validate_empty_container(self):
    """validate does not raise an error for a container with no records."""
    reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nE')
    reader.validate()