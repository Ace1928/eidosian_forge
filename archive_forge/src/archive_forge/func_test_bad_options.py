import json
import os
from os.path import join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import (
def test_bad_options(self):
    with ReaderFactory('sol') as reader:
        if reader is None:
            raise IOError("Reader 'sol' is not registered")
        with self.assertRaises(ValueError):
            soln = reader(join(currdir, 'bad_options.sol'))