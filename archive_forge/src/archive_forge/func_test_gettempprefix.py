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
def test_gettempprefix(self):
    ctx = self.TM.new_context()
    pre = ctx.gettempprefix()
    self.assertIsInstance(pre, str)
    self.assertEqual(pre, tempfile.gettempprefix())
    preb = ctx.gettempprefixb()
    self.assertIsInstance(preb, bytes)
    self.assertEqual(preb, tempfile.gettempprefixb())