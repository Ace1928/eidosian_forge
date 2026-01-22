import enum
import textwrap
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.expr import EqualityExpression
from pyomo.util.subsystems import create_subsystem_block
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import (
from pyomo.common.deprecation import deprecated
from pyomo.contrib.incidence_analysis.config import get_config_from_kwds
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.connected import get_independent_submatrices
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import (
from pyomo.contrib.incidence_analysis.incidence import get_incident_variables
from pyomo.contrib.pynumero.asl import AmplInterface
def get_structural_incidence_matrix(variables, constraints, **kwds):
    """Return the incidence matrix of Pyomo constraints and variables

    Parameters
    ---------
    variables: List of Pyomo VarData objects
    constraints: List of Pyomo ConstraintData objects
    include_fixed: Bool
        Flag for whether fixed variables should be included in the matrix
        nonzeros

    Returns
    -------
    ``scipy.sparse.coo_matrix``
        COO matrix. Rows are indices into the user-provided list of constraints,
        columns are indices into the user-provided list of variables.
        Entries are 1.0.

    """
    config = get_config_from_kwds(**kwds)
    _check_unindexed(variables + constraints)
    N, M = (len(variables), len(constraints))
    var_idx_map = ComponentMap(((v, i) for i, v in enumerate(variables)))
    rows = []
    cols = []
    for i, con in enumerate(constraints):
        cols.extend((var_idx_map[v] for v in get_incident_variables(con.body, **config) if v in var_idx_map))
        rows.extend([i] * (len(cols) - len(rows)))
    assert len(rows) == len(cols)
    data = [1.0] * len(rows)
    matrix = sp.sparse.coo_matrix((data, (rows, cols)), shape=(M, N))
    return matrix