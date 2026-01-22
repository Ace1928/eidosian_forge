from __future__ import annotations
import os
import zipfile
from typing import Union
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.zippath import ZipArchive, ZipPath
from twisted.test.test_paths import AbstractFilePathTests
def test_zipPathRepr(self) -> None:
    """
        Make sure that invoking ZipPath's repr prints the correct class name
        and an absolute path to the zip file.
        """
    child: Union[ZipPath[str, bytes], ZipPath[str, str]] = self.path.child('foo')
    pathRepr = 'ZipPath({!r})'.format(os.path.abspath(self.nativecmn + '.zip' + os.sep + 'foo'))
    self.assertEqual(repr(child), pathRepr)
    relativeCommon = self.nativecmn.replace(os.getcwd() + os.sep, '', 1) + '.zip'
    relpath = ZipArchive(relativeCommon)
    child = relpath.child('foo')
    self.assertEqual(repr(child), pathRepr)