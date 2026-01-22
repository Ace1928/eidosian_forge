from collections.abc import Iterable
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect, gurobipy
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import value, is_fixed
from pyomo.opt.base import SolverFactory
def set_var_attr(self, var, attr, val):
    """
        Set the value of an attribute on a gurobi variable.

        Parameters
        ----------
        con: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding gurobi var attribute
            should be modified.
        attr: str
            The attribute to be modified. Options are:

                Start
                VarHintVal
                VarHintPri
                BranchPriority
                VBasis
                PStart

        val: any
            See gurobi documentation for acceptable values.
        """
    if attr in {'LB', 'UB', 'VType', 'VarName'}:
        raise ValueError('Var attr {0} cannot be set with' + ' the set_var_attr method. Please use' + ' the update_var method.'.format(attr))
    if attr == 'Obj':
        raise ValueError('Var attr Obj cannot be set with' + ' the set_var_attr method. Please use' + ' the set_objective method.')
    if self._version_major < 7:
        if self._solver_model.getAttr('NumVars') == 0 or self._solver_model.getVarByName(self._symbol_map.getSymbol(var)) is None:
            self._solver_model.update()
    self._pyomo_var_to_solver_var_map[var].setAttr(attr, val)
    self._needs_updated = True