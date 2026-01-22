from pyomo.core.expr.numvalue import value
from pyomo.solvers.plugins.solvers.cplex_direct import CPLEXDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base import SolverFactory
def update_var(self, var):
    """Update a single variable in the solver's model.

        This will update bounds, fix/unfix the variable as needed, and
        update the variable type.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)

        """
    if var not in self._pyomo_var_to_solver_var_map:
        raise ValueError('The Var provided to compile_var needs to be added first: {0}'.format(var))
    cplex_var = self._pyomo_var_to_solver_var_map[var]
    vtype = self._cplex_vtype_from_var(var)
    lb, ub = self._cplex_lb_ub_from_var(var)
    self._solver_model.variables.set_lower_bounds(cplex_var, lb)
    self._solver_model.variables.set_upper_bounds(cplex_var, ub)
    self._solver_model.variables.set_types(cplex_var, vtype)