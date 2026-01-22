from collections.abc import Iterable
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect, gurobipy
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import value, is_fixed
from pyomo.opt.base import SolverFactory
def set_linear_constraint_attr(self, con, attr, val):
    """
        Set the value of an attribute on a gurobi linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be modified.
        attr: str
            The attribute to be modified. Options are:

                CBasis
                DStart
                Lazy

        val: any
            See gurobi documentation for acceptable values.
        """
    if attr in {'Sense', 'RHS', 'ConstrName'}:
        raise ValueError('Linear constraint attr {0} cannot be set with' + ' the set_linear_constraint_attr method. Please use' + ' the remove_constraint and add_constraint methods.'.format(attr))
    if self._version_major < 7:
        if self._solver_model.getAttr('NumConstrs') == 0 or self._solver_model.getConstrByName(self._symbol_map.getSymbol(con)) is None:
            self._solver_model.update()
    self._pyomo_con_to_solver_con_map[con].setAttr(attr, val)
    self._needs_updated = True