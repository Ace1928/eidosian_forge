from io import BytesIO
from ... import tests
from .. import pack
def test_validate_data_after_end_marker(self):
    """validate raises ContainerHasExcessDataError if there are any bytes
        after the end of the container.
        """
    reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nEcrud')
    self.assertRaises(pack.ContainerHasExcessDataError, reader.validate)