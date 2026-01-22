import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
def test_parameter_estimation_example(self):
    from pyomo.contrib.parmest.examples.reactor_design import parameter_estimation_example
    parameter_estimation_example.main()