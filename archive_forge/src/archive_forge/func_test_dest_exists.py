import io
import os
import shutil
import sys
import tempfile
from dulwich.tests import SkipTest, TestCase
from ..file import FileLocked, GitFile, _fancy_rename
def test_dest_exists(self):
    self.create(self.bar, b'bar contents')
    _fancy_rename(self.foo, self.bar)
    self.assertFalse(os.path.exists(self.foo))
    new_f = open(self.bar, 'rb')
    self.assertEqual(b'foo contents', new_f.read())
    new_f.close()