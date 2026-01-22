from io import BytesIO
from ... import tests
from .. import pack
def test_record_with_no_name(self):
    """Reading a Bytes record with no name returns an empty list of
        names.
        """
    self.assertRecordParsing(([], b'aaaaa'), b'5\n\naaaaa')