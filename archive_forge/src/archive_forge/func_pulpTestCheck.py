import os
import tempfile
from pulp.constants import PulpError
from pulp.apis import *
from pulp import LpVariable, LpProblem, lpSum, LpConstraintVar, LpFractionConstraint
from pulp import constants as const
from pulp.tests.bin_packing_problem import create_bin_packing_problem
from pulp.utilities import makeDict
import functools
import unittest
def pulpTestCheck(prob, solver, okstatus, sol=None, reducedcosts=None, duals=None, slacks=None, eps=10 ** (-3), status=None, objective=None, **kwargs):
    if status is None:
        status = prob.solve(solver, **kwargs)
    if status not in okstatus:
        dumpTestProblem(prob)
        raise PulpError('Tests failed for solver {}:\nstatus == {} not in {}\nstatus == {} not in {}'.format(solver, status, okstatus, const.LpStatus[status], [const.LpStatus[s] for s in okstatus]))
    if sol is not None:
        for v, x in sol.items():
            if abs(v.varValue - x) > eps:
                dumpTestProblem(prob)
                raise PulpError('Tests failed for solver {}:\nvar {} == {} != {}'.format(solver, v, v.varValue, x))
    if reducedcosts:
        for v, dj in reducedcosts.items():
            if abs(v.dj - dj) > eps:
                dumpTestProblem(prob)
                raise PulpError('Tests failed for solver {}:\nTest failed: var.dj {} == {} != {}'.format(solver, v, v.dj, dj))
    if duals:
        for cname, p in duals.items():
            c = prob.constraints[cname]
            if abs(c.pi - p) > eps:
                dumpTestProblem(prob)
                raise PulpError('Tests failed for solver {}:\nconstraint.pi {} == {} != {}'.format(solver, cname, c.pi, p))
    if slacks:
        for cname, slack in slacks.items():
            c = prob.constraints[cname]
            if abs(c.slack - slack) > eps:
                dumpTestProblem(prob)
                raise PulpError('Tests failed for solver {}:\nconstraint.slack {} == {} != {}'.format(solver, cname, c.slack, slack))
    if objective is not None:
        z = prob.objective.value()
        if abs(z - objective) > eps:
            dumpTestProblem(prob)
            raise PulpError(f'Tests failed for solver {solver}:\nobjective {z} != {objective}')