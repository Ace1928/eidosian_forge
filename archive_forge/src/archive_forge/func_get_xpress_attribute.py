from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.solvers.plugins.solvers.xpress_direct import XpressDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.numvalue import value, is_fixed
import pyomo.core.expr as EXPR
from pyomo.opt.base import SolverFactory
import collections
def get_xpress_attribute(self, *args):
    """
        Get xpress attributes.

        Parameters
        ----------
        control(s): str, strs, list, None
            The xpress attribute to get. Options include any xpress attribute.
            Can also be list of xpress controls or None for every attribute
            Please see the Xpress documentation for options.

        See the Xpress documentation for xpress.problem.getAttrib for other
        uses of this function

        Returns
        -------
        control value or dictionary of control values
        """
    return self._solver_model.getAttrib(*args)