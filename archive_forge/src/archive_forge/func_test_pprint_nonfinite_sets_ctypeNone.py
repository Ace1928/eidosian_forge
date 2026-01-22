import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.common.collections import ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set import (
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import (
def test_pprint_nonfinite_sets_ctypeNone(self):
    self.maxDiff = None
    m = ConcreteModel()
    m.v = Var(NonNegativeIntegers, dense=False)
    m.ref = Reference(m.v, ctype=None)
    buf = StringIO()
    m.pprint(ostream=buf)
    self.assertEqual(buf.getvalue().strip(), '\n1 Var Declarations\n    v : Size=0, Index=NonNegativeIntegers\n        Key : Lower : Value : Upper : Fixed : Stale : Domain\n\n1 IndexedComponent Declarations\n    ref : Size=0, Index=NonNegativeIntegers, ReferenceTo=v\n        Key : Object\n\n2 Declarations: v ref\n'.strip())
    m.v[3]
    m.ref[5]
    buf = StringIO()
    m.pprint(ostream=buf)
    self.assertEqual(buf.getvalue().strip(), "\n1 Var Declarations\n    v : Size=2, Index=NonNegativeIntegers\n        Key : Lower : Value : Upper : Fixed : Stale : Domain\n          3 :  None :  None :  None : False :  True :  Reals\n          5 :  None :  None :  None : False :  True :  Reals\n\n1 IndexedComponent Declarations\n    ref : Size=2, Index=NonNegativeIntegers, ReferenceTo=v\n        Key : Object\n          3 : <class 'pyomo.core.base.var._GeneralVarData'>\n          5 : <class 'pyomo.core.base.var._GeneralVarData'>\n\n2 Declarations: v ref\n".strip())