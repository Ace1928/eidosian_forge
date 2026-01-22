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
def test_nondata_bounds(self):
    A = numpy.ones((5, 4))
    ctuple = matrix_constraint(A, rhs=1)
    eL = expression()
    eU = expression()
    with self.assertRaises(ValueError):
        ctuple.rhs = eL
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.rhs = eL
    self.assertTrue((ctuple.rhs == 1).all())
    self.assertTrue((ctuple.equality == True).all())
    vL = variable()
    vU = variable()
    with self.assertRaises(ValueError):
        ctuple.rhs = vL
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.rhs = vL
    self.assertTrue((ctuple.rhs == 1).all())
    self.assertTrue((ctuple.equality == True).all())
    vL.value = 1.0
    vU.value = 1.0
    with self.assertRaises(ValueError):
        ctuple.rhs = vL
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.rhs = vL
    self.assertTrue((ctuple.rhs == 1).all())
    self.assertTrue((ctuple.equality == True).all())
    vL.fixed = True
    vU.fixed = True
    with self.assertRaises(ValueError):
        ctuple.rhs = vL
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.rhs = vL
    self.assertTrue((ctuple.rhs == 1).all())
    self.assertTrue((ctuple.equality == True).all())
    p = parameter(value=0)
    with self.assertRaises(ValueError):
        ctuple.rhs = p
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.rhs = p
    self.assertTrue((ctuple.rhs == 1).all())
    self.assertTrue((ctuple.equality == True).all())
    ctuple.equality = False
    self.assertTrue((ctuple.lb == 1).all())
    self.assertTrue((ctuple.ub == 1).all())
    self.assertTrue((ctuple.equality == False).all())
    eL = expression()
    eU = expression()
    with self.assertRaises(ValueError):
        ctuple.lb = eL
    with self.assertRaises(ValueError):
        ctuple.ub = eU
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.lb = eL
        with self.assertRaises(ValueError):
            c.ub = eU
        with self.assertRaises(ValueError):
            c.bounds = (eL, eU)
    self.assertTrue((ctuple.lb == 1).all())
    self.assertTrue((ctuple.ub == 1).all())
    self.assertTrue((ctuple.equality == False).all())
    vL = variable()
    vU = variable()
    with self.assertRaises(ValueError):
        ctuple.lb = vL
    with self.assertRaises(ValueError):
        ctuple.ub = vU
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.lb = vL
        with self.assertRaises(ValueError):
            c.ub = vU
        with self.assertRaises(ValueError):
            c.bounds = (vL, vU)
    self.assertTrue((ctuple.lb == 1).all())
    self.assertTrue((ctuple.ub == 1).all())
    self.assertTrue((ctuple.equality == False).all())
    vL.value = 1.0
    vU.value = 1.0
    with self.assertRaises(ValueError):
        ctuple.lb = vL
    with self.assertRaises(ValueError):
        ctuple.ub = vU
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.lb = vL
        with self.assertRaises(ValueError):
            c.ub = vU
        with self.assertRaises(ValueError):
            c.bounds = (vL, vU)
    self.assertTrue((ctuple.lb == 1).all())
    self.assertTrue((ctuple.ub == 1).all())
    self.assertTrue((ctuple.equality == False).all())
    vL.fixed = True
    vU.fixed = True
    with self.assertRaises(ValueError):
        ctuple.lb = vL
    with self.assertRaises(ValueError):
        ctuple.ub = vU
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.lb = vL
        with self.assertRaises(ValueError):
            c.ub = vU
        with self.assertRaises(ValueError):
            c.bounds = (vL, vU)
    self.assertTrue((ctuple.lb == 1).all())
    self.assertTrue((ctuple.ub == 1).all())
    self.assertTrue((ctuple.equality == False).all())
    p = parameter(value=0)
    with self.assertRaises(ValueError):
        ctuple.lb = p
    with self.assertRaises(ValueError):
        ctuple.ub = p
    for c in ctuple:
        with self.assertRaises(ValueError):
            c.lb = p
        with self.assertRaises(ValueError):
            c.ub = p
        with self.assertRaises(ValueError):
            c.bounds = (p, p)
    self.assertTrue((ctuple.lb == 1).all())
    self.assertTrue((ctuple.ub == 1).all())
    self.assertTrue((ctuple.equality == False).all())