from io import BytesIO
from ... import tests
from .. import pack
def test_record_with_one_name(self):
    """Reading a Bytes record with one name returns a list of just that
        name.
        """
    self.assertRecordParsing(([(b'name1',)], b'aaaaa'), b'5\nname1\n\naaaaa')