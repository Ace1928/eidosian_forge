from __future__ import print_function
import unittest
import numpy as np
import cvxpy as cvx
import cvxpy.interface as intf
from cvxpy.reductions.solvers.conic_solvers import ecos_conif
from cvxpy.tests.base_test import BaseTest
def test_advanced1(self) -> None:
    """Code from the advanced tutorial.
        """
    x = cvx.Variable(2)
    obj = cvx.Minimize(x[0] + cvx.norm(x, 1))
    constraints = [x >= 2]
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.ECOS)
    print('optimal value with ECOS:', prob.value)
    self.assertAlmostEqual(prob.value, 6)
    if cvx.CVXOPT in cvx.installed_solvers():
        prob.solve(solver=cvx.CVXOPT)
        print('optimal value with CVXOPT:', prob.value)
        self.assertAlmostEqual(prob.value, 6)
    prob.solve(solver=cvx.SCS)
    print('optimal value with SCS:', prob.value)
    self.assertAlmostEqual(prob.value, 6, places=2)
    if cvx.CPLEX in cvx.installed_solvers():
        prob.solve(solver=cvx.CPLEX)
        print('optimal value with CPLEX:', prob.value)
        self.assertAlmostEqual(prob.value, 6)
    if cvx.GLPK in cvx.installed_solvers():
        prob.solve(solver=cvx.GLPK)
        print('optimal value with GLPK:', prob.value)
        self.assertAlmostEqual(prob.value, 6)
        prob.solve(solver=cvx.GLPK_MI)
        print('optimal value with GLPK_MI:', prob.value)
        self.assertAlmostEqual(prob.value, 6)
    if cvx.GUROBI in cvx.installed_solvers():
        prob.solve(solver=cvx.GUROBI)
        print('optimal value with GUROBI:', prob.value)
        self.assertAlmostEqual(prob.value, 6)
    if cvx.XPRESS in cvx.installed_solvers():
        prob.solve(solver=cvx.XPRESS)
        print('optimal value with XPRESS:', prob.value)
        self.assertAlmostEqual(prob.value, 6)
    print(cvx.installed_solvers())