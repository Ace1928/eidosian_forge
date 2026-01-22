import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_disc_second_order_central(self):
    m = self.m.clone()
    m.dv1dt2 = DerivativeVar(m.v1, wrt=(m.t, m.t))
    disc = TransformationFactory('dae.finite_difference')
    disc.apply_to(m, nfe=2, scheme='CENTRAL')
    self.assertTrue(hasattr(m, 'dv1dt2_disc_eq'))
    self.assertEqual(len(m.dv1dt2_disc_eq), 1)
    self.assertEqual(len(m.v1), 3)
    output = 'dv1dt2_disc_eq : Size=1, Index=t, Active=True\n    Key : Lower : Body                                            : Upper : Active\n    5.0 :   0.0 : dv1dt2[5.0] - 0.04*(v1[10] - 2*v1[5.0] + v1[0]) :   0.0 :   True\n'
    out = StringIO()
    m.dv1dt2_disc_eq.pprint(ostream=out)
    self.assertEqual(output, out.getvalue())