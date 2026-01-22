import re
import pyomo.common.unittest as unittest
from pyomo.common.tee import capture_output
from pyomo.common.log import LoggingIntercept
from pyomo.environ import SolverFactory
from pyomo.scripting.driver_help import help_solvers, help_transformations
from pyomo.scripting.pyomo_main import main
def test_pyomo_main_deprecation(self):
    with LoggingIntercept() as LOG:
        with unittest.pytest.raises(SystemExit) as e:
            main(args=['--solvers=glpk', 'foo.py'])
    self.assertIn("Running the 'pyomo' script with no subcommand", LOG.getvalue())