from io import BytesIO
from ... import tests
from .. import pack
def test_container_has_excess_data(self):
    """Test the formatting of ContainerHasExcessDataError."""
    e = pack.ContainerHasExcessDataError('excess bytes')
    self.assertEqual("Container has data after end marker: 'excess bytes'", str(e))