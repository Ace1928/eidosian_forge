import collections
import itertools
import math
import numpy as np
import numpy.linalg as LA
import pytest
import cvxpy as cp
import cvxpy.interface as intf
from cvxpy.error import SolverError
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.settings import CVXOPT, ECOS, MOSEK, OSQP, ROBUST_KKTSOLVER, SCS
@pytest.mark.parametrize('atom_info, objective_type', atoms_minimize + atoms_maximize)
def test_constant_atoms(atom_info, objective_type) -> None:
    atom, size, args, obj_val = atom_info
    for indexer in get_indices(size):
        for solver in SOLVERS_TO_TRY:
            prob_val = obj_val[indexer].value
            const_args = [Constant(arg) for arg in args]
            if len(size) != 0:
                objective = objective_type(atom(*const_args)[indexer])
            else:
                objective = objective_type(atom(*const_args))
            problem = Problem(objective)
            run_atom(atom, problem, prob_val, solver)
            variables = []
            constraints = []
            for idx, expr in enumerate(args):
                variables.append(Variable(intf.shape(expr)))
                constraints.append(variables[-1] == expr)
            if len(size) != 0:
                objective = objective_type(atom(*variables)[indexer])
            else:
                objective = objective_type(atom(*variables))
            problem = Problem(objective, constraints)
            run_atom(atom, problem, prob_val, solver)
            parameters = []
            for expr in args:
                parameters.append(Parameter(intf.shape(expr)))
                parameters[-1].value = intf.DEFAULT_INTF.const_to_matrix(expr)
            if len(size) != 0:
                objective = objective_type(atom(*parameters)[indexer])
            else:
                objective = objective_type(atom(*parameters))
            run_atom(atom, Problem(objective), prob_val, solver)