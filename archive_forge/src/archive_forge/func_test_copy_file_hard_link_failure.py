import unittest
import os
import errno
from unittest.mock import patch
from distutils.file_util import move_file, copy_file
from distutils import log
from distutils.tests import support
from distutils.errors import DistutilsFileError
from test.support.os_helper import unlink
@unittest.skipUnless(hasattr(os, 'link'), 'requires os.link')
def test_copy_file_hard_link_failure(self):
    with open(self.source, 'w') as f:
        f.write('some content')
    st = os.stat(self.source)
    with patch('os.link', side_effect=OSError(0, 'linking unsupported')):
        copy_file(self.source, self.target, link='hard')
    st2 = os.stat(self.source)
    st3 = os.stat(self.target)
    self.assertTrue(os.path.samestat(st, st2), (st, st2))
    self.assertFalse(os.path.samestat(st2, st3), (st2, st3))
    for fn in (self.source, self.target):
        with open(fn, 'r') as f:
            self.assertEqual(f.read(), 'some content')