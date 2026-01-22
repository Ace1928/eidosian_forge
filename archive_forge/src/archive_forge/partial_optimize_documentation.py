from typing import List, Optional, Tuple
import cvxpy.settings as s
import cvxpy.utilities as u
from cvxpy.atoms import sum, trace
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem
Returns the graph implementation of the object.

        Change the ids of all the opt_vars.

        Returns
        -------
            A tuple of (affine expression, [constraints]).
        