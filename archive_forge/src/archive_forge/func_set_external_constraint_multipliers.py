import itertools
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.timing import HierarchicalTimer
from pyomo.util.subsystems import create_subsystem_block
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
import numpy as np
import scipy.sparse as sps
def set_external_constraint_multipliers(self, eq_con_multipliers):
    eq_con_multipliers = np.array(eq_con_multipliers)
    external_multipliers = self.calculate_external_constraint_multipliers(eq_con_multipliers)
    multipliers = np.concatenate((eq_con_multipliers, external_multipliers))
    cons = self.residual_cons + self.external_cons
    n_con = len(cons)
    assert n_con == self._nlp.n_constraints()
    duals = np.zeros(n_con)
    indices = self._nlp.get_constraint_indices(cons)
    duals[indices] = multipliers
    self._nlp.set_duals(duals)