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
@unittest.skipIf(NamedTuple is None, 'typing module not available')
def test_issue_938(self):
    self.maxDiff = None
    NodeKey = NamedTuple('NodeKey', [('id', int)])
    ArcKey = NamedTuple('ArcKey', [('node_from', NodeKey), ('node_to', NodeKey)])

    def build_model():
        model = ConcreteModel()
        model.node_keys = Set(doc='Set of nodes', initialize=[NodeKey(0), NodeKey(1)])
        model.arc_keys = Set(doc='Set of arcs', within=model.node_keys * model.node_keys, initialize=[ArcKey(NodeKey(0), NodeKey(0)), ArcKey(NodeKey(0), NodeKey(1))])
        model.arc_variables = Var(model.arc_keys, within=Binary)

        def objective_rule(model_arg):
            return sum((var for var in model_arg.arc_variables.values()))
        model.obj = Objective(rule=objective_rule)
        return model
    try:
        _oldFlatten = normalize_index.flatten
        normalize_index.flatten = True
        m = build_model()
        output = StringIO()
        m.pprint(ostream=output)
        ref = '\n2 Set Declarations\n    arc_keys : Set of arcs\n        Size=1, Index=None, Ordered=Insertion\n        Key  : Dimen : Domain              : Size : Members\n        None :     2 : node_keys*node_keys :    2 : {(0, 0), (0, 1)}\n    node_keys : Set of nodes\n        Size=1, Index=None, Ordered=Insertion\n        Key  : Dimen : Domain : Size : Members\n        None :     1 :    Any :    2 : {0, 1}\n\n1 Var Declarations\n    arc_variables : Size=2, Index=arc_keys\n        Key    : Lower : Value : Upper : Fixed : Stale : Domain\n        (0, 0) :     0 :  None :     1 : False :  True : Binary\n        (0, 1) :     0 :  None :     1 : False :  True : Binary\n\n1 Objective Declarations\n    obj : Size=1, Index=None, Active=True\n        Key  : Active : Sense    : Expression\n        None :   True : minimize : arc_variables[0,0] + arc_variables[0,1]\n\n4 Declarations: node_keys arc_keys arc_variables obj\n'.strip()
        self.assertEqual(output.getvalue().strip(), ref)
        normalize_index.flatten = False
        m = build_model()
        output = StringIO()
        m.pprint(ostream=output)
        ref = '\n2 Set Declarations\n    arc_keys : Set of arcs\n        Size=1, Index=None, Ordered=Insertion\n        Key  : Dimen : Domain              : Size : Members\n        None :     2 : node_keys*node_keys :    2 : {ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=0)), ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=1))}\n    node_keys : Set of nodes\n        Size=1, Index=None, Ordered=Insertion\n        Key  : Dimen : Domain : Size : Members\n        None :     1 :    Any :    2 : {NodeKey(id=0), NodeKey(id=1)}\n\n1 Var Declarations\n    arc_variables : Size=2, Index=arc_keys\n        Key                                                    : Lower : Value : Upper : Fixed : Stale : Domain\n        ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=0)) :     0 :  None :     1 : False :  True : Binary\n        ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=1)) :     0 :  None :     1 : False :  True : Binary\n\n1 Objective Declarations\n    obj : Size=1, Index=None, Active=True\n        Key  : Active : Sense    : Expression\n        None :   True : minimize : arc_variables[ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=0))] + arc_variables[ArcKey(node_from=NodeKey(id=0), node_to=NodeKey(id=1))]\n\n4 Declarations: node_keys arc_keys arc_variables obj\n'.strip()
        self.assertEqual(output.getvalue().strip(), ref)
    finally:
        normalize_index.flatten = _oldFlatten