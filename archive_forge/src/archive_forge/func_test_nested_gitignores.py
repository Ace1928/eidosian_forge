import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
def test_nested_gitignores(self):
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    repo = Repo.init(tmp_dir)
    with open(os.path.join(repo.path, '.gitignore'), 'wb') as f:
        f.write(b'/*\n')
        f.write(b'!/foo\n')
    os.mkdir(os.path.join(repo.path, 'foo'))
    with open(os.path.join(repo.path, 'foo', '.gitignore'), 'wb') as f:
        f.write(b'/bar\n')
    with open(os.path.join(repo.path, 'foo', 'bar'), 'wb') as f:
        f.write(b'IGNORED')
    m = IgnoreFilterManager.from_repo(repo)
    self.assertTrue(m.is_ignored('foo/bar'))