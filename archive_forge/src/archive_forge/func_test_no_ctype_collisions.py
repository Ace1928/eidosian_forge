import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.variable import IVariable
from pyomo.core.kernel.constraint import IConstraint
def test_no_ctype_collisions(self):
    hash_set = set()
    hash_list = list()
    for cls in [pmo.variable, pmo.constraint, pmo.objective, pmo.expression, pmo.parameter, pmo.suffix, pmo.sos, pmo.block]:
        ctype = cls._ctype
        hash_set.add(hash(ctype))
        hash_list.append(hash(ctype))
    self.assertEqual(len(hash_set), len(hash_list))