from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipUnless(platform.isWindows(), 'special rename EIO handling only necessary and correct on Windows.')
def test_symlinkEIOWindows(self) -> None:
    """
        L{lockfile.symlink} raises L{OSError} with C{errno} set to L{EIO} when
        the underlying L{rename} call fails with L{EIO}.

        Renaming a file on Windows may fail if the target of the rename is in
        the process of being deleted (directory deletion appears not to be
        atomic).
        """
    name = self.mktemp()

    def fakeRename(src: str, dst: str) -> NoReturn:
        raise OSError(errno.EIO, None)
    self.patch(lockfile, 'rename', fakeRename)
    exc = self.assertRaises(IOError, lockfile.symlink, name, 'foo')
    self.assertEqual(exc.errno, errno.EIO)