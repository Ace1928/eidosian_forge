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
def test_solver_resolvable_invalid_type(self):
    """
        Test solver resolvable object raises expected
        exception when invalid entry is provided.
        """
    invalid_object = 2
    standardizer_func = SolverResolvable(solver_desc='local solver')
    exc_str = 'Cannot cast object `2` to a Pyomo optimizer.*local solver.*got type int.*'
    with self.assertRaisesRegex(SolverNotResolvable, exc_str):
        standardizer_func(invalid_object)