from itertools import zip_longest
import json
import os
import sys
from os.path import join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
def test3_write_mps(self):
    """Convert from AMPL to MPS"""
    if not pyomo.common.Executable('ampl'):
        self.skipTest('The ampl executable is not available')
    self.model = pyomo.opt.AmplModel(join(currdir, 'test3.mod'))
    _test = TempfileManager.create_tempfile(suffix='test3.mps')
    try:
        self.model.write(_test)
    except ApplicationError:
        err = sys.exc_info()[1]
        if pyomo.common.Executable('ampl'):
            self.fail("Unexpected ApplicationError - ampl is enabled but not available: '%s'" % str(err))
        return
    except pyomo.opt.ConverterError:
        err = sys.exc_info()[1]
        if pyomo.common.Executable('ampl'):
            self.fail("Unexpected ConverterError - ampl is enabled but not available: '%s'" % str(err))
        return
    _base = join(currdir, 'test3.baseline.mps')
    with open(_test, 'r') as run, open(_base, 'r') as baseline:
        for line1, line2 in zip_longest(run, baseline):
            for _pattern in ('NAME',):
                if line1.find(_pattern) >= 0:
                    line1 = line1[:line1.find(_pattern) + len(_pattern)]
                    line2 = line2[:line2.find(_pattern) + len(_pattern)]
            self.assertEqual(line1, line2, msg='Files %s and %s differ' % (_test, _base))