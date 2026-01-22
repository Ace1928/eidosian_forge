from io import BytesIO
from ... import tests
from .. import pack
def test_add_bytes_record_split_writes(self):
    """Write a large record which does multiple IOs"""
    writes = []
    real_write = self.writer.write_func

    def record_writes(data):
        writes.append(data)
        return real_write(data)
    self.writer.write_func = record_writes
    self.writer._JOIN_WRITES_THRESHOLD = 2
    self.writer.begin()
    offset, length = self.writer.add_bytes_record([b'abcabc'], len(b'abcabc'), names=[(b'name1',)])
    self.assertEqual((42, 16), (offset, length))
    self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nB6\nname1\n\nabcabc')
    self.assertEqual([b'Bazaar pack format 1 (introduced in 0.18)\n', b'B6\nname1\n\n', b'abcabc'], writes)