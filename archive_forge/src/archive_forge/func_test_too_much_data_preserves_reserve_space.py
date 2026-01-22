import zlib
from .. import chunk_writer
from . import TestCaseWithTransport
def test_too_much_data_preserves_reserve_space(self):
    lines = self._make_lines()
    writer = chunk_writer.ChunkWriter(4096, 256)
    for idx, line in enumerate(lines):
        if writer.write(line):
            self.assertEqual(44, idx)
            break
    else:
        self.fail('We were able to write all lines')
    self.assertFalse(writer.write(b'A' * 256, reserved=True))
    bytes_list, unused, _ = writer.finish()
    node_bytes = self.check_chunk(bytes_list, 4096)
    expected_bytes = b''.join(lines[:44]) + b'A' * 256
    self.assertEqualDiff(expected_bytes, node_bytes)
    self.assertEqual(lines[44], unused)