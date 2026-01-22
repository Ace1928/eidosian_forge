from io import BytesIO
from ... import tests
from .. import pack
def test_validate_non_empty_valid_container(self):
    """validate does not raise an error for a container with a valid record.
        """
    reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nB3\nname\n\nabcE')
    reader.validate()