import gc
import glob
import os
import shutil
import sys
import tempfile
from io import StringIO
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.common.tempfiles as tempfiles
from pyomo.common.dependencies import pyutilib_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import (
def test_create_tempdir(self):
    context = self.TM.push()
    fname = self.TM.create_tempdir('suffix', 'prefix')
    self.assertRegex(os.path.basename(fname), '^prefix')
    self.assertRegex(os.path.basename(fname), 'suffix$')
    self.assertGreater(len(os.path.basename(fname)), len('prefixsuffix'))
    self.assertTrue(os.path.exists(fname))
    self.assertTrue(os.path.isdir(fname))
    context.release()
    self.assertFalse(os.path.exists(fname))