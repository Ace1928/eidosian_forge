import logging
import unittest
from pyomo.core.base import ConcreteModel, Var, _VarData
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import ApplicationError
from pyomo.core.base.param import Param, _ParamData
from pyomo.contrib.pyros.config import (
from pyomo.contrib.pyros.util import ObjectiveType
from pyomo.opt import SolverFactory, SolverResults
from pyomo.contrib.pyros.uncertainty_sets import BoxSet
from pyomo.common.dependencies import numpy_available
def test_solver_resolvable_valid_str(self):
    """
        Test solver resolvable class is valid for string
        type.
        """
    solver_str = AVAILABLE_SOLVER_TYPE_NAME
    standardizer_func = SolverResolvable()
    solver = standardizer_func(solver_str)
    expected_solver_type = type(SolverFactory(solver_str))
    self.assertIsInstance(solver, type(SolverFactory(solver_str)), msg=f'SolverResolvable object should be of type {expected_solver_type.__name__}, but got object of type {solver.__class__.__name__}.')