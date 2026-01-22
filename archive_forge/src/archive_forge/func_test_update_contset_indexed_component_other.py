import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_update_contset_indexed_component_other(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.junk = Suffix()
    m.s = Set(initialize=[1, 2, 3])
    m.v = Var(m.s)

    def _obj(m):
        return sum((m.v[i] for i in m.s))
    m.obj = Objective(rule=_obj)
    expansion_map = ComponentMap
    generate_finite_elements(m.t, 5)
    update_contset_indexed_component(m.junk, expansion_map)
    update_contset_indexed_component(m.s, expansion_map)
    update_contset_indexed_component(m.obj, expansion_map)