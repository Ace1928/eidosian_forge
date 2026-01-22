import pickle
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.tuple_container import TupleContainer
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.matrix_constraint import matrix_constraint, _MatrixConstraintData
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression
from pyomo.core.kernel.block import block, block_list
def test_slack(self):
    vlist = _create_variable_list(3)
    vlist[0].value = 1
    vlist[1].value = 0
    vlist[2].value = 3
    A = numpy.ones((3, 3))
    ctuple = matrix_constraint(A, x=vlist)
    self.assertTrue((ctuple() == 4).all())
    self.assertEqual(ctuple[0](), 4)
    self.assertEqual(ctuple[1](), 4)
    self.assertEqual(ctuple[2](), 4)
    A[:, 0] = 0
    A[:, 2] = 2
    ctuple = matrix_constraint(A, x=vlist)
    vlist[2].value = 4
    self.assertTrue((ctuple() == 8).all())
    self.assertEqual(ctuple[0](), 8)
    self.assertEqual(ctuple[1](), 8)
    self.assertEqual(ctuple[2](), 8)
    A = numpy.random.rand(4, 3)
    ctuple = matrix_constraint(A, x=vlist)
    vlist[1].value = 2
    cvals = numpy.array([ctuple[0](), ctuple[1](), ctuple[2](), ctuple[3]()])
    self.assertTrue((ctuple() == cvals).all())