import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
def test_uncertain_bounds_to_constraints(self):
    m = ConcreteModel()
    m.p = Param(initialize=8, mutable=True)
    m.r = Param(initialize=-5, mutable=True)
    m.q = Param(initialize=1, mutable=False)
    m.s = Param(initialize=1, mutable=True)
    m.n = Param(initialize=1, mutable=True)
    m.u = Var(initialize=0, bounds=(0, m.p))
    m.v = Var(initialize=1, bounds=(m.r, m.p))
    m.w = Var(initialize=1, bounds=(None, None))
    m.x = Var(initialize=1, bounds=(0, exp(-1 * m.p / 8) * m.q * m.s))
    m.y = Var(initialize=-1, bounds=(m.r * m.p, 0))
    m.z = Var(initialize=1, bounds=(0, m.s))
    m.t = Var(initialize=1, bounds=(0, m.p ** 2))
    m.obj = Objective(sense=maximize, expr=m.x ** 2 - m.y + m.t ** 2 + m.v)
    mod = m.clone()
    uncertain_params = [mod.n, mod.p, mod.r]
    mod.obj.deactivate()
    replace_uncertain_bounds_with_constraints(mod, uncertain_params)
    self.assertTrue(hasattr(mod, 'uncertain_var_bound_cons'), msg='Uncertain variable bounds erroneously added. Check only variables participating in active objective and constraints are added.')
    self.assertFalse(mod.uncertain_var_bound_cons)
    mod.obj.activate()
    constraints_m = ConstraintList()
    m.add_component('perf_constraints', constraints_m)
    constraints_m.add(m.w == 2 * m.x + m.y)
    constraints_m.add(m.v + m.x + m.y >= 0)
    constraints_m.add(m.y ** 2 + m.z >= 0)
    constraints_m.add(m.x ** 2 + m.u <= 1)
    constraints_m[4].deactivate()
    mod_2 = m.clone()
    uncertain_cons = ConstraintList()
    m.add_component('uncertain_var_bound_cons', uncertain_cons)
    uncertain_cons.add(m.x - m.x.upper <= 0)
    uncertain_cons.add(m.y.lower - m.y <= 0)
    uncertain_cons.add(m.v - m.v._ub <= 0)
    uncertain_cons.add(m.v.lower - m.v <= 0)
    uncertain_cons.add(m.t - m.t.upper <= 0)
    m.x.setub(None)
    m.y.setlb(None)
    m.v.setlb(None)
    m.v.setub(None)
    m.t.setub(None)
    svars_con = ComponentSet(get_vars_from_component(mod_2, Constraint))
    svars_obj = ComponentSet(get_vars_from_component(mod_2, Objective))
    vars_in_active_cons = ComponentSet([mod_2.z, mod_2.w, mod_2.y, mod_2.x, mod_2.v])
    vars_in_active_obj = ComponentSet([mod_2.x, mod_2.y, mod_2.t, mod_2.v])
    self.assertEqual(svars_con, vars_in_active_cons, msg='Mismatch of variables participating in activated constraints.')
    self.assertEqual(svars_obj, vars_in_active_obj, msg='Mismatch of variables participating in activated objectives.')
    uncertain_params = [mod_2.p, mod_2.r]
    replace_uncertain_bounds_with_constraints(mod_2, uncertain_params)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), len(list(mod_2.component_data_objects(Constraint))), msg='Mismatch between number of explicit variable bound inequality constraints added automatically and added manually.')
    vars_in_cons = ComponentSet()
    params_in_cons = ComponentSet()
    cons = mod_2.uncertain_var_bound_cons
    for idx in cons:
        for p in identify_mutable_parameters(cons[idx].expr):
            params_in_cons.add(p)
        for v in identify_variables(cons[idx].expr):
            vars_in_cons.add(v)
    params_in_cons = params_in_cons & uncertain_params
    vars_with_bounds_removed = ComponentSet([mod_2.x, mod_2.y, mod_2.v, mod_2.t])
    self.assertEqual(params_in_cons, ComponentSet([mod_2.p, mod_2.r]), msg='Mismatch of parameters added to explicit inequality constraints.')
    self.assertEqual(vars_in_cons, vars_with_bounds_removed, msg='Mismatch of variables added to explicit inequality constraints.')