import numpy as np
import abc
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.environ import Var, Constraint, value
from pyomo.core.base.var import _VarData
from pyomo.common.modeling import unique_component_name
def load_x_into_pyomo(self, primals):
    """
        Use this method to load a numpy array of values into the corresponding
        Pyomo variables (e.g., the solution from CyIpopt)

        Parameters
        ----------
        primals : numpy array
           The array of values that will be given to the Pyomo variables. The
           order of this array is the same as the order in the PyomoNLP created
           internally.
        """
    pyomo_variables = self._pyomo_nlp.get_pyomo_variables()
    for i, v in enumerate(primals):
        pyomo_variables[i].set_value(v)