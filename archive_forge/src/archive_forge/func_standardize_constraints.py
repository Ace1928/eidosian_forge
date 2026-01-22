from warnings import warn
import numpy as np
from ._optimize import (_minimize_neldermead, _minimize_powell, _minimize_cg,
from ._trustregion_dogleg import _minimize_dogleg
from ._trustregion_ncg import _minimize_trust_ncg
from ._trustregion_krylov import _minimize_trust_krylov
from ._trustregion_exact import _minimize_trustregion_exact
from ._trustregion_constr import _minimize_trustregion_constr
from ._lbfgsb_py import _minimize_lbfgsb
from ._tnc import _minimize_tnc
from ._cobyla_py import _minimize_cobyla
from ._slsqp_py import _minimize_slsqp
from ._constraints import (old_bound_to_new, new_bounds_to_old,
from ._differentiable_functions import FD_METHODS
def standardize_constraints(constraints, x0, meth):
    """Converts constraints to the form required by the solver."""
    all_constraint_types = (NonlinearConstraint, LinearConstraint, dict)
    new_constraint_types = all_constraint_types[:-1]
    if constraints is None:
        constraints = []
    elif isinstance(constraints, all_constraint_types):
        constraints = [constraints]
    else:
        constraints = list(constraints)
    if meth in ['trust-constr', 'new']:
        for i, con in enumerate(constraints):
            if not isinstance(con, new_constraint_types):
                constraints[i] = old_constraint_to_new(i, con)
    else:
        for i, con in enumerate(list(constraints)):
            if isinstance(con, new_constraint_types):
                old_constraints = new_constraint_to_old(con, x0)
                constraints[i] = old_constraints[0]
                constraints.extend(old_constraints[1:])
    return constraints