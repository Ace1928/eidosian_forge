import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, SOSConstraint, Var, Set
def test14(self):

    def rule(model, i):
        if i == 0:
            return SOSConstraint.Skip
        else:
            return (list((M.x[i] for i in M.x)), [1, 20, 3])
    w = {0: {1: 10, 2: 2, 3: 30}, 1: {1: 1, 2: 20, 3: 3}}
    M = ConcreteModel()
    M.x = Var([1, 2, 3], dense=True)
    M.c = SOSConstraint([0, 1], rule=rule, sos=1)
    self.assertEqual(list(M.c.keys()), [1])
    self.assertEqual(set(((v.name, w) for v, w in M.c[1].get_items())), set(((M.x[i].name, w[1][i]) for i in [1, 2, 3])))