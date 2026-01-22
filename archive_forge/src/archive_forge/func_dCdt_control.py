import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib
def dCdt_control(m, y, t):
    """
            Calculate CA in Jacobian matrix analytically
            y: CA, CB, CC
            t: timepoints
            """
    if y == 'CA':
        return m.dCdt[y, t] == -m.kp1[t] * m.C['CA', t]
    elif y == 'CB':
        return m.dCdt[y, t] == m.kp1[t] * m.C['CA', t] - m.kp2[t] * m.C['CB', t]
    elif y == 'CC':
        return pyo.Constraint.Skip