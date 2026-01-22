from pyomo.common.dependencies import (
import pyomo.common.unittest as unittest
from pyomo.opt import SolverFactory
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not scipy_available, 'scipy is not available')
@unittest.skipIf(not numpy_available, 'Numpy is not available')
def test_reactor_compute_FIM(self):
    from pyomo.contrib.doe.examples import reactor_compute_FIM
    reactor_compute_FIM.main()