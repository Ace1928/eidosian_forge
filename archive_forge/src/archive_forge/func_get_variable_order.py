import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def get_variable_order(self, vartype=None):
    """
        This function returns the ordered list of differential variable
        names. The order corresponds to the order being sent to the
        integrator function. Knowing the order allows users to provide
        initial conditions for the differential equations using a
        list or map the profiles returned by the simulate function to
        the Pyomo variables.

        Parameters
        ----------
        vartype : `string` or None
            Optional argument for specifying the type of variables to return
            the order for. The default behavior is to return the order of
            the differential variables. 'time-varying' will return the order
            of all the time-dependent algebraic variables identified in the
            model. 'algebraic' will return the order of algebraic variables
            used in the most recent call to the simulate function. 'input'
            will return the order of the time-dependent algebraic variables
            that were treated as inputs in the most recent call to the
            simulate function.

        Returns
        -------
        `list`

        """
    if vartype == 'time-varying':
        return self._algvars
    elif vartype == 'algebraic':
        return self._simalgvars
    elif vartype == 'input':
        return self._siminputvars
    else:
        return self._diffvars