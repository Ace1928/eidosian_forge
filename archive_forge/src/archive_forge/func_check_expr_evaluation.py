from io import StringIO
import shlex
from tempfile import mkdtemp
import os, sys, math, logging, shutil, time, subprocess
from pyomo.core.base import Constraint, Var, value, Objective
from pyomo.opt import ProblemFormat, SolverFactory
import pyomo.common
from pyomo.common.collections import Bunch
from pyomo.common.tee import TeeStream
from pyomo.opt.base.solvers import _extract_version
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.objective import IObjective
from pyomo.core.kernel.variable import IVariable
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.opt.results import (
from pyomo.common.dependencies import attempt_import
def check_expr_evaluation(model, symbolMap, solver_io):
    try:
        uninit_vars = list()
        for var in model.component_data_objects(Var):
            if var.value is None:
                uninit_vars.append(var)
                var.set_value(0, skip_validation=True)
        for con in model.component_data_objects(Constraint, active=True):
            if con.body.is_fixed():
                continue
            check_expr(con.body, con.name, solver_io)
        obj = list(model.component_data_objects(Objective, active=True))
        assert len(obj) == 1, 'GAMS writer can only take 1 active objective'
        obj = obj[0]
        check_expr(obj.expr, obj.name, solver_io)
    finally:
        for var in uninit_vars:
            var.set_value(None)