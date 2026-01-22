from io import BytesIO
from ... import tests
from .. import pack
def test_multiple_empty_records_at_once(self):
    """If multiple empty records worth of data are fed to the parser in one
        string, the parser will correctly parse all the records.

        (A naive implementation might stop after parsing the first empty
        record, because the buffer size had not changed.)
        """
    parser = self.make_parser_expecting_record_type()
    parser.accept_bytes(b'B0\nname1\n\nB0\nname2\n\n')
    self.assertEqual([([(b'name1',)], b''), ([(b'name2',)], b'')], parser.read_pending_records())