import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
def evaluate_jacobian_outputs(self):
    delta = 1e-06
    u0 = np.copy(self._input_values)
    y0 = self.evaluate_outputs()
    jac = np.empty((4, 5))
    u = np.copy(self._input_values)
    for j in range(len(u)):
        u[j] += delta
        self.set_input_values(u)
        yperturb = self.evaluate_outputs()
        jac_col = (yperturb - y0) / delta
        jac[:, j] = jac_col
        u[j] = u0[j]
    self.set_input_values(u0)
    row = []
    col = []
    data = []
    for r in range(4):
        for c in range(5):
            row.append(r)
            col.append(c)
            data.append(jac[r, c])
    return coo_matrix((data, (row, col)), shape=(4, 5))