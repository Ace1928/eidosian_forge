import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
def test_reader_registration(self):
    """
        Testing methods in the reader factory registration process
        """
    ReaderFactory.unregister('rtest3')
    self.assertTrue(not 'rtest3' in ReaderFactory)
    ReaderFactory.register('rtest3')(MockReader)
    self.assertTrue('rtest3' in ReaderFactory)