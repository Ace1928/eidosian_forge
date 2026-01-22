import os
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.mpec import Complementarity, complements, ComplementarityList
from pyomo.opt import ProblemFormat
from pyomo.repn.plugins.nl_writer import FileDeterminism
from pyomo.repn.tests.nl_diff import load_and_compare_nl_baseline
def test_cov10(self):
    M = self._setup()
    M.cc = Complementarity(expr=complements(inequality(M.y, M.x1, 1), M.x2))
    try:
        M.cc.to_standard_form()
        self.fail('Expected a RuntimeError')
    except RuntimeError:
        pass