import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
def test_writer_factory(self):
    """
        Testing the pyomo.opt writer factory with MIP writers
        """
    WriterFactory.register('wtest3')(MockWriter)
    factory = WriterFactory
    self.assertTrue(set(['wtest3']) <= set(factory))