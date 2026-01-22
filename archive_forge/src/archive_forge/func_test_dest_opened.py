import io
import os
import shutil
import sys
import tempfile
from dulwich.tests import SkipTest, TestCase
from ..file import FileLocked, GitFile, _fancy_rename
def test_dest_opened(self):
    if sys.platform != 'win32':
        raise SkipTest('platform allows overwriting open files')
    self.create(self.bar, b'bar contents')
    dest_f = open(self.bar, 'rb')
    self.assertRaises(OSError, _fancy_rename, self.foo, self.bar)
    dest_f.close()
    self.assertTrue(os.path.exists(self.path('foo')))
    new_f = open(self.foo, 'rb')
    self.assertEqual(b'foo contents', new_f.read())
    new_f.close()
    new_f = open(self.bar, 'rb')
    self.assertEqual(b'bar contents', new_f.read())
    new_f.close()