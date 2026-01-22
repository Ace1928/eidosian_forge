import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
import scipy.sparse as spa
import numpy as np
import math
def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
    assert len(eq_con_multiplier_values) == 6
    np.copyto(self._eq_constraint_multipliers, eq_con_multiplier_values)