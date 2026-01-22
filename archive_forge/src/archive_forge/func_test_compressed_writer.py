from gitdb.test.lib import (
from gitdb import (
from gitdb.util import hex_to_bin
import zlib
from gitdb.typ import (
import tempfile
import os
from io import BytesIO
def test_compressed_writer(self):
    for ds in self.data_sizes:
        fd, path = tempfile.mkstemp()
        ostream = FDCompressedSha1Writer(fd)
        data = make_bytes(ds, randomize=False)
        assert len(data) == ostream.write(data)
        ostream.close()
        self.assertRaises(OSError, os.close, fd)
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
        written_data = os.read(fd, os.path.getsize(path))
        assert len(written_data) == os.path.getsize(path)
        os.close(fd)
        assert written_data == zlib.compress(data, 1)
        os.remove(path)