from io import BytesIO
from ... import tests
from .. import pack
def test_duplicate_record_name_error(self):
    """Test the formatting of DuplicateRecordNameError."""
    e = pack.DuplicateRecordNameError(b'n\xc3\xa5me')
    self.assertEqual('Container has multiple records with the same name: nåme', str(e))