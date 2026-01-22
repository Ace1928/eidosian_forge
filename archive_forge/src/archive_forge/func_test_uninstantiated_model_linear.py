import re
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.scripting.pyomo_main import main
from pyomo.core import (
from pyomo.common.tee import capture_output
from io import StringIO
def test_uninstantiated_model_linear(self):
    """Run pyomo with "bad" model file.  Should fail gracefully, with
        a perhaps useful-to-the-user message."""
    if not 'glpk' in solvers:
        self.skipTest('glpk solver is not available')
    return
    base = '%s/test_uninstantiated_model' % currdir
    fout, fbase = (join(base, '_linear.out'), join(base, '.txt'))
    self.pyomo('uninstantiated_model_linear.py', file=fout)
    self.assertTrue(cmp(fout, fbase), msg='Files %s and %s differ' % (fout, fbase))