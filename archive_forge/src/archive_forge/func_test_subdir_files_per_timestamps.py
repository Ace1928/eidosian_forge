import gzip
import os
import tarfile
import time
import zipfile
from io import BytesIO
from .. import errors, export, tests
from ..archive.tar import tarball_generator
from ..export import get_root_name
from . import features
def test_subdir_files_per_timestamps(self):
    builder = self.make_branch_builder('source')
    builder.start_series()
    foo_time = time.mktime((1999, 12, 12, 0, 0, 0, 0, 0, 0))
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('subdir', b'subdir-id', 'directory', '')), ('add', ('subdir/foo.txt', b'foo-id', 'file', b'content\n'))], timestamp=foo_time)
    builder.finish_series()
    b = builder.get_branch()
    b.lock_read()
    self.addCleanup(b.unlock)
    tree = b.basis_tree()
    export.export(tree, 'target', format='dir', subdir='subdir', per_file_timestamps=True)
    t = self.get_transport('target')
    self.assertEqual(foo_time, t.stat('foo.txt').st_mtime)