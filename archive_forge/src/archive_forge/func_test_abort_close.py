import io
import os
import shutil
import sys
import tempfile
from dulwich.tests import SkipTest, TestCase
from ..file import FileLocked, GitFile, _fancy_rename
def test_abort_close(self):
    foo = self.path('foo')
    f = GitFile(foo, 'wb')
    f.abort()
    try:
        f.close()
    except OSError:
        self.fail()
    f = GitFile(foo, 'wb')
    f.close()
    try:
        f.abort()
    except OSError:
        self.fail()