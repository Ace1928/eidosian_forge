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
def test_files_same_timestamp(self):
    builder = self.make_branch_builder('source')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('a', b'a-id', 'file', b'content\n'))])
    builder.build_snapshot(None, [('add', ('b', b'b-id', 'file', b'content\n'))])
    builder.finish_series()
    b = builder.get_branch()
    b.lock_read()
    self.addCleanup(b.unlock)
    tree = b.basis_tree()
    orig_iter_files_bytes = tree.iter_files_bytes

    def iter_files_bytes(to_fetch):
        for thing in orig_iter_files_bytes(to_fetch):
            yield thing
            time.sleep(1)
    tree.iter_files_bytes = iter_files_bytes
    export.export(tree, 'target', format='dir')
    t = self.get_transport('target')
    st_a = t.stat('a')
    st_b = t.stat('b')
    self.assertEqual(st_a.st_mtime, st_b.st_mtime)