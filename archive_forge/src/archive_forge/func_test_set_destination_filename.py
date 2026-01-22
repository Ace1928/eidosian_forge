import os
import platform
import re
import shutil
import tempfile
import subprocess
import pyomo.common.unittest as unittest
import pyomo.common.envvar as envvar
from pyomo.common import DeveloperError
from pyomo.common.fileutils import this_file
from pyomo.common.download import FileDownloader, distro_available
from pyomo.common.tee import capture_output
def test_set_destination_filename(self):
    self.tmpdir = os.path.abspath(tempfile.mkdtemp())
    f = FileDownloader()
    self.assertIsNone(f._fname)
    f.set_destination_filename('foo')
    self.assertEqual(f._fname, os.path.join(envvar.PYOMO_CONFIG_DIR, 'foo'))
    self.assertTrue(os.path.isdir(envvar.PYOMO_CONFIG_DIR))
    f.target = self.tmpdir
    f.set_destination_filename('foo')
    target = os.path.join(self.tmpdir, 'foo')
    self.assertEqual(f._fname, target)
    self.assertFalse(os.path.exists(target))
    f.target = self.tmpdir
    f.set_destination_filename(os.path.join('foo', 'bar'))
    target = os.path.join(self.tmpdir, 'foo', 'bar')
    self.assertEqual(f._fname, target)
    self.assertFalse(os.path.exists(target))
    target_dir = os.path.join(self.tmpdir, 'foo')
    self.assertTrue(os.path.isdir(target_dir))