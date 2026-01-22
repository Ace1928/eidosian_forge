import sys
import unittest
import os
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
def test_get_vc2015(self):
    import distutils._msvccompiler as _msvccompiler
    version, path = _msvccompiler._find_vc2015()
    if version:
        self.assertGreaterEqual(version, 14)
        self.assertTrue(os.path.isdir(path))
    else:
        raise unittest.SkipTest('VS 2015 is not installed')