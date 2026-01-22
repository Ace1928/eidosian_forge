import zlib
from .. import chunk_writer
from . import TestCaseWithTransport
def test_optimize_for_speed(self):
    writer = chunk_writer.ChunkWriter(4096)
    writer.set_optimize(for_size=False)
    self.assertEqual(chunk_writer.ChunkWriter._repack_opts_for_speed, (writer._max_repack, writer._max_zsync))
    writer = chunk_writer.ChunkWriter(4096, optimize_for_size=False)
    self.assertEqual(chunk_writer.ChunkWriter._repack_opts_for_speed, (writer._max_repack, writer._max_zsync))