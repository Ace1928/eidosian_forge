import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
def bounding_model(self, config=None):
    """
        Make uncertain parameter value bounding problems (optimize
        value of each uncertain parameter subject to constraints on the
        uncertain parameters).

        Parameters
        ----------
        config : None or ConfigDict, optional
            If a ConfigDict is provided, then it contains
            arguments passed to the PyROS solver.

        Returns
        -------
        model : ConcreteModel
            Bounding problem, with all Objectives deactivated.
        """
    model = ConcreteModel()
    model.util = Block()
    model.param_vars = Var(range(self.dim))
    model.cons = self.set_as_constraint(uncertain_params=model.param_vars, model=model, config=config)

    @model.Objective(range(self.dim))
    def param_var_objectives(self, idx):
        return model.param_vars[idx]
    model.param_var_objectives.deactivate()
    return model