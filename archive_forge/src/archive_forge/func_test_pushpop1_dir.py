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
def test_pushpop1_dir(self):
    """Test pushpop logic with directories"""
    TempfileManager.push()
    os.mkdir(tempdir + 'pushpop1')
    TempfileManager.add_tempfile(tempdir + 'pushpop1')
    TempfileManager.pop()
    if os.path.exists(tempdir + 'pushpop1'):
        self.fail('pop() failed to clean out directories')