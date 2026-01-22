from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.dependencies import attempt_import, numpy as np
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.scc_solver import (
Partitions the systems of equations defined by the provided
        variables and constraints

        Each subset of the partition should have an equal number of variables
        and equations. These subsets, or "subsystems", will be solved
        sequentially in the order provided by this method instead of solving
        the entire system simultaneously. Subclasses should implement this
        method to define the partition that their implicit function solver
        will use. Partitions are defined as a list of tuples of lists.
        Each tuple has two entries, the first a list of variables, and the
        second a list of constraints. These inner lists should have the
        same number of entries.

        Arguments
        ---------
        variables: list
            List of VarData in the system to be partitioned
        constraints: list
            List of ConstraintData (equality constraints) defining the
            equations of the system to be partitioned

        Returns
        -------
        List of tuples
            List of tuples describing the ordered partition. Each tuple
            contains equal-length subsets of variables and constraints.

        