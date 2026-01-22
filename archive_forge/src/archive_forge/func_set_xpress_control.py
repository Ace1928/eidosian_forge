from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.solvers.plugins.solvers.xpress_direct import XpressDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.numvalue import value, is_fixed
import pyomo.core.expr as EXPR
from pyomo.opt.base import SolverFactory
import collections
def set_xpress_control(self, *args):
    """
        Set xpress controls.

        Parameters
        ----------
        control: str
            The xpress control to set. Options include any xpree control.
            Please see the Xpress documentation for options.
        val: any
            The value to set the control to. See Xpress documentation for possible values.

        If one argument, it must be a dictionary with control keys and control values
        """
    self._solver_model.setControl(*args)