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
def check_standard_form(self, m):
    compBlock = m.disjunct1.component('comp')
    self.assertIsInstance(compBlock, Block)
    self.assertIsInstance(compBlock.component('v'), Var)
    self.assertIsInstance(compBlock.component('c'), Constraint)
    self.assertIsInstance(compBlock.component('ve'), Constraint)