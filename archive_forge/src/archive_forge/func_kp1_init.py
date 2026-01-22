import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib
def kp1_init(m, t):
    return m.A1 * pyo.exp(-m.E1 * 1000 / (m.R * m.T[t]))