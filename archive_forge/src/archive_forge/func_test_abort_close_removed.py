import io
import os
import shutil
import sys
import tempfile
from dulwich.tests import SkipTest, TestCase
from ..file import FileLocked, GitFile, _fancy_rename
def test_abort_close_removed(self):
    foo = self.path('foo')
    f = GitFile(foo, 'wb')
    f._file.close()
    os.remove(foo + '.lock')
    f.abort()
    self.assertTrue(f._closed)