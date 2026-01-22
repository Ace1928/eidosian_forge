from itertools import zip_longest
import re
import sys
import os
from os.path import join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import ProblemFormat, ConverterError, convert_problem
from pyomo.common import Executable
@unittest.skipUnless(Executable('glpsol').available(), 'glpsol required')
def test_mod_lp2(self):
    ans = convert_problem((join(currdir, 'test5.mod'), join(currdir, 'test5.dat')), None, [ProblemFormat.cpxlp])
    self.assertTrue(ans[0][0].endswith('glpsol.lp'))
    with open(ans[0][0], 'r') as f1, open(join(currdir, 'test3_convert.lp'), 'r') as f2:
        for line1, line2 in zip_longest(f1, f2):
            if 'Problem' in line1:
                continue
            self.assertEqual(line1, line2)