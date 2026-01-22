from collections.abc import Iterable
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect, gurobipy
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import value, is_fixed
from pyomo.opt.base import SolverFactory
def get_var_attr(self, var, attr):
    """
        Get the value of an attribute on a gurobi var.

        Parameters
        ----------
        var: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding gurobi var attribute
            should be retrieved.
        attr: str
            The attribute to get. Options are:

                LB
                UB
                Obj
                VType
                VarName
                X
                Xn
                RC
                BarX
                Start
                VarHintVal
                VarHintPri
                BranchPriority
                VBasis
                PStart
                IISLB
                IISUB
                PWLObjCvx
                SAObjLow
                SAObjUp
                SALBLow
                SALBUp
                SAUBLow
                SAUBUp
                UnbdRay
        """
    if self._needs_updated:
        self._update()
    return self._pyomo_var_to_solver_var_map[var].getAttr(attr)