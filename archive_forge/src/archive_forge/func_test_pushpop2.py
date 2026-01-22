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
def test_pushpop2(self):
    """Test pushpop logic"""
    TempfileManager.push()
    OUTPUT = open(tempdir + 'pushpop2', 'w')
    OUTPUT.write('tempfile\n')
    OUTPUT.close()
    TempfileManager.add_tempfile(tempdir + 'pushpop2')
    TempfileManager.push()
    OUTPUT = open(tempdir + 'pushpop2a', 'w')
    OUTPUT.write('tempfile\n')
    OUTPUT.close()
    TempfileManager.add_tempfile(tempdir + 'pushpop2a')
    TempfileManager.pop()
    if not os.path.exists(tempdir + 'pushpop2'):
        self.fail('pop() clean out all files')
    if os.path.exists(tempdir + 'pushpop2a'):
        self.fail('pop() failed to clean out files')
    TempfileManager.pop()
    if os.path.exists(tempdir + 'pushpop2'):
        self.fail('pop() failed to clean out files')