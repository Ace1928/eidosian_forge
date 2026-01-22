import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_disc_single_index_central(self):
    m = self.m.clone()
    disc = TransformationFactory('dae.finite_difference')
    disc.apply_to(m, nfe=5, scheme='CENTRAL')
    self.assertTrue(hasattr(m, 'dv1_disc_eq'))
    self.assertEqual(len(m.dv1_disc_eq), 4)
    self.assertEqual(len(m.v1), 6)
    expected_disc_points = [0, 2.0, 4.0, 6.0, 8.0, 10]
    disc_info = m.t.get_discretization_info()
    self.assertEqual(disc_info['scheme'], 'CENTRAL Difference')
    for idx, val in enumerate(list(m.t)):
        self.assertAlmostEqual(val, expected_disc_points[idx])
    output = 'dv1_disc_eq : Size=4, Index=t, Active=True\n    Key : Lower : Body                                : Upper : Active\n    2.0 :   0.0 :   dv1[2.0] - 0.25*(v1[4.0] - v1[0]) :   0.0 :   True\n    4.0 :   0.0 : dv1[4.0] - 0.25*(v1[6.0] - v1[2.0]) :   0.0 :   True\n    6.0 :   0.0 : dv1[6.0] - 0.25*(v1[8.0] - v1[4.0]) :   0.0 :   True\n    8.0 :   0.0 :  dv1[8.0] - 0.25*(v1[10] - v1[6.0]) :   0.0 :   True\n'
    out = StringIO()
    m.dv1_disc_eq.pprint(ostream=out)
    self.assertEqual(output, out.getvalue())