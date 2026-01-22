from io import BytesIO
from ... import tests
from .. import pack
def test_non_empty_end_does_not_add_a_record_to_records_written(self):
    """The end() method does not count towards the records written."""
    self.writer.begin()
    self.writer.add_bytes_record([b'foo'], len(b'foo'), names=[])
    self.writer.end()
    self.assertEqual(1, self.writer.records_written)