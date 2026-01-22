import sys
import unittest
import os
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
def test_remove_visual_c_ref(self):
    from distutils.msvc9compiler import MSVCCompiler
    tempdir = self.mkdtemp()
    manifest = os.path.join(tempdir, 'manifest')
    f = open(manifest, 'w')
    try:
        f.write(_MANIFEST_WITH_MULTIPLE_REFERENCES)
    finally:
        f.close()
    compiler = MSVCCompiler()
    compiler._remove_visual_c_ref(manifest)
    f = open(manifest)
    try:
        content = '\n'.join([line.rstrip() for line in f.readlines()])
    finally:
        f.close()
    self.assertEqual(content, _CLEANED_MANIFEST)