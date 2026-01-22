import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
def test_multisensor_data_example(self):
    from pyomo.contrib.parmest.examples.reactor_design import multisensor_data_example
    multisensor_data_example.main()