import zlib
from .. import chunk_writer
from . import TestCaseWithTransport
def test_too_much_data_does_not_exceed_size(self):
    lines = self._make_lines()
    writer = chunk_writer.ChunkWriter(4096)
    for idx, line in enumerate(lines):
        if writer.write(line):
            self.assertEqual(46, idx)
            break
    bytes_list, unused, _ = writer.finish()
    node_bytes = self.check_chunk(bytes_list, 4096)
    expected_bytes = b''.join(lines[:46])
    self.assertEqualDiff(expected_bytes, node_bytes)
    self.assertEqual(lines[46], unused)