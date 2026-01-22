import unittest
import sys
import os
from io import BytesIO
from distutils import cygwinccompiler
from distutils.cygwinccompiler import (check_config_h,
from distutils.tests import support
def test_get_versions(self):
    self.assertEqual(get_versions(), (None, None, None))
    self._exes['gcc'] = b'gcc (GCC) 3.4.5 (mingw special)\nFSF'
    res = get_versions()
    self.assertEqual(str(res[0]), '3.4.5')
    self._exes['gcc'] = b'very strange output'
    res = get_versions()
    self.assertEqual(res[0], None)
    self._exes['ld'] = b'GNU ld version 2.17.50 20060824'
    res = get_versions()
    self.assertEqual(str(res[1]), '2.17.50')
    self._exes['ld'] = b'@(#)PROGRAM:ld  PROJECT:ld64-77'
    res = get_versions()
    self.assertEqual(res[1], None)
    self._exes['dllwrap'] = b'GNU dllwrap 2.17.50 20060824\nFSF'
    res = get_versions()
    self.assertEqual(str(res[2]), '2.17.50')
    self._exes['dllwrap'] = b'Cheese Wrap'
    res = get_versions()
    self.assertEqual(res[2], None)