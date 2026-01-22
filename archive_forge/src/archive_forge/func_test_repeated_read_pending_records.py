from io import BytesIO
from ... import tests
from .. import pack
def test_repeated_read_pending_records(self):
    """read_pending_records will not return the same record twice."""
    parser = self.make_parser_expecting_bytes_record()
    parser.accept_bytes(b'6\n\nabcdef')
    self.assertEqual([([], b'abcdef')], parser.read_pending_records())
    self.assertEqual([], parser.read_pending_records())