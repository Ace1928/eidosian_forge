from pyomo.common.dependencies import (
import pyomo.common.unittest as unittest
from pyomo.opt import SolverFactory
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not pandas_available, 'pandas is not available')
@unittest.skipIf(not numpy_available, 'Numpy is not available')
def test_reactor_grid_search(self):
    from pyomo.contrib.doe.examples import reactor_grid_search
    reactor_grid_search.main()