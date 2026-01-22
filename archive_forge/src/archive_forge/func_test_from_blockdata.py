import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def test_from_blockdata(self):
    m = self.model()
    context = m.b.b1[3].b.b2[2, 'b', 2]
    comp = m.b.b1[3].b.b2[2, 'b', 2].b
    pred_stack = [(self.get_attribute, 'b')]
    self.assertCorrectStack(comp, pred_stack, context=context)
    comp = m.b.b1[3].b.b2[2, 'b', 2].b.v2['a', 1]
    pred_stack = [(self.get_item, ('a', 1)), (self.get_attribute, 'v2'), (self.get_attribute, 'b')]
    self.assertCorrectStack(comp, pred_stack, context=context)
    context = m.b.b1[3]
    comp = m.b.b1[3]
    act_stack = get_component_call_stack(comp, context=context)
    self.assertEqual(len(act_stack), 0)