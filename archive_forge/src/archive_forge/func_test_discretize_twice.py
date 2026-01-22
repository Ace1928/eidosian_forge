import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_discretize_twice(self):
    m = self.m.clone()
    disc1 = TransformationFactory('dae.finite_difference')
    disc1.apply_to(m, nfe=5)
    disc2 = TransformationFactory('dae.finite_difference')
    with self.assertRaises(DAE_Error):
        disc2.apply_to(m, nfe=5)