import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
def test_domain_and_pprint(self):
    m = ConcreteModel()
    m.I = SetOf([1, 2])
    m.A = m.I * [3, 4]
    self.assertIs(m.A._domain, m.A)
    m.A._domain = Any
    self.assertIs(m.A._domain, m.A)
    with self.assertRaisesRegex(ValueError, 'Setting the domain of a Set Operator is not allowed'):
        m.A._domain = None
    output = StringIO()
    m.A.pprint(ostream=output)
    ref = '\nA : Size=1, Index=None, Ordered=True\n    Key  : Dimen : Domain   : Size : Members\n    None :     2 : I*{3, 4} :    4 : {(1, 3), (1, 4), (2, 3), (2, 4)}\n'.strip()
    self.assertEqual(output.getvalue().strip(), ref)
    m = ConcreteModel()
    m.I = Set(initialize=[1, 2, 3])
    m.J = Reals * m.I
    output = StringIO()
    m.J.pprint(ostream=output)
    ref = '\nJ : Size=1, Index=None, Ordered=False\n    Key  : Dimen : Domain  : Size : Members\n    None :     2 : Reals*I :  Inf : <[-inf..inf], ([1], [2], [3])>\n'.strip()
    self.assertEqual(output.getvalue().strip(), ref)