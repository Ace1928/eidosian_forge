import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
@b.Block(m.s1, m.s2)
def b2(b, i1, i2, i3=None):
    b.v0 = pyo.Var()
    b.v1 = pyo.Var(m.s1)

    @b.Block()
    def b(b):
        b.v0 = pyo.Var()
        b.v2 = pyo.Var(m.s2)