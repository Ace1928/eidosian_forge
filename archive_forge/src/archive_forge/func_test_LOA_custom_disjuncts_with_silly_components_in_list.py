from contextlib import redirect_stdout
from io import StringIO
import logging
from math import fabs
from os.path import join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.contrib.appsi.solvers.gurobi import Gurobi
from pyomo.contrib.gdpopt.create_oa_subproblems import (
import pyomo.contrib.gdpopt.tests.common_tests as ct
from pyomo.contrib.gdpopt.util import is_feasible, time_code
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.tests import models
from pyomo.opt import TerminationCondition
def test_LOA_custom_disjuncts_with_silly_components_in_list(self):
    exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
    eight_process = exfile.build_eight_process_flowsheet()
    eight_process.goofy = Disjunct()
    eight_process.goofy.deactivate()
    initialize = [[eight_process.use_unit_1or2.disjuncts[0], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[1], eight_process.use_unit_8ornot.disjuncts[0], eight_process.goofy], [eight_process.use_unit_1or2.disjuncts[1], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[0], eight_process.use_unit_8ornot.disjuncts[0]]]
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
        results = SolverFactory('gdpopt.loa').solve(eight_process, init_algorithm='custom_disjuncts', custom_init_disjuncts=initialize, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertIn('The following disjuncts from the custom disjunct initialization set number 0 were unused: goofy', output.getvalue().strip())
    self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 0.01)