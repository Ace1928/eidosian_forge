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
def test_tgz_consistent_mtime(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    wt.add(['a'])
    timestamp = 1547400500
    revid = wt.commit('1', timestamp=timestamp)
    revtree = wt.branch.repository.revision_tree(revid)
    export.export(revtree, 'target.tar.gz', format='tgz')
    with gzip.GzipFile('target.tar.gz', 'r') as f:
        f.read()
        self.assertEqual(int(f.mtime), timestamp)