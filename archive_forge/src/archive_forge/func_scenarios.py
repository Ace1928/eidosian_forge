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
@scenarios.setter
def scenarios(self, val):
    validate_array(arr=val, arr_name='scenarios', dim=2, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
    scenario_arr = np.array(val)
    if hasattr(self, '_scenarios'):
        if scenario_arr.shape[1] != self.dim:
            raise ValueError(f"DiscreteScenarioSet attribute 'scenarios' must have {self.dim} columns to match set dimension (provided array-like with {scenario_arr.shape[1]} columns)")
    self._scenarios = [tuple(s) for s in val]