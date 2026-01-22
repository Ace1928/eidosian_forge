import numpy as np
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
@property
def ineq_form(self) -> bool:
    """
        Choose between two constraining methodologies, use ``ineq_form=False`` while
        working with ``Parameter`` types.
        """
    return self._ineq_form