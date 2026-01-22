import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_indexed_block_mutable_param(self):
    model = AbstractModel()
    model.A = RangeSet(2)

    def _b_rule(b, id):
        b.A = Param(initialize=id, mutable=True)
    model.B = Block(model.A, rule=_b_rule)
    instance = model.create_instance()
    self.assertEqual(value(instance.B[1].A), 1)
    self.assertEqual(value(instance.B[2].A), 2)
    instance.B[1].A = 4.0
    self.assertEqual(value(instance.B[1].A), 4.0)