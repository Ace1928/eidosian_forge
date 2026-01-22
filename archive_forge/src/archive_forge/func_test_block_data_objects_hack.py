import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.variable import IVariable
from pyomo.core.kernel.constraint import IConstraint
def test_block_data_objects_hack(self):
    model = _model.clone()
    model.deactivate()
    self.assertEqual([str(obj) for obj in model.block_data_objects(active=True)], [])
    self.assertEqual([str(obj) for obj in model.block_data_objects()], [str(model)] + [str(obj) for obj in model.components(ctype=IBlock)])
    model.activate()
    self.assertEqual([str(obj) for obj in model.block_data_objects(active=True)], [str(model)] + [str(obj) for obj in model.components(ctype=IBlock)])
    self.assertEqual([str(obj) for obj in model.block_data_objects()], [str(model)] + [str(obj) for obj in model.components(ctype=IBlock)])