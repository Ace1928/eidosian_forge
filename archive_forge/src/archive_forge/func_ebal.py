import pyomo.environ as pyo
import pyomo.dae as dae
def ebal(m, i):
    if i == 0:
        return pyo.Constraint.Skip
    else:
        return m.rho[i - 1] * m.F[i - 1] * m.T[i - 1] + m.Q[i] - m.rho[i] * m.F[i] * m.T[i] == 0