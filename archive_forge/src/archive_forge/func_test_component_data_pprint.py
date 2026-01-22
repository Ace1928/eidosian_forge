from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.environ import (
from pyomo.core.base.set import GlobalSets
def test_component_data_pprint(self):
    m = ConcreteModel()
    m.a = Set(initialize=[1, 2, 3], ordered=True)
    m.x = Var(m.a)
    stream = StringIO()
    m.x[2].pprint(ostream=stream)
    correct_s = '{Member of x} : Size=3, Index=a\n    Key : Lower : Value : Upper : Fixed : Stale : Domain\n      2 :  None :  None :  None : False :  True :  Reals\n'
    self.assertEqual(correct_s, stream.getvalue())