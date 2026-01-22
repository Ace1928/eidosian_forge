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
def test_pyomo_nl1(self):
    ans = convert_problem((join(currdir, 'model.py'), ProblemFormat.nl), None, [ProblemFormat.nl])
    self.assertNotEqual(re.match('.*tmp.*pyomo.nl$', ans[0][0]), None)
    os.remove(ans[0][0])