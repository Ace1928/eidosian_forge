import pyomo.environ as pyo
import pyomo.dae as dae
def ideal_gas(m, i):
    return m.P[i] - m.rho[i] * m.R * m.T[i] == 0