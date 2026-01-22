from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
from pyomo.common.dependencies.matplotlib import pyplot as plt
def plot_optimal_solution(m):
    SolverFactory('ipopt').solve(m, tee=True)
    x = []
    u = []
    F = []
    for ii in m.t:
        x.append(value(m.x[ii]))
        u.append(value(m.u[ii]))
        F.append(value(m.F[ii]))
    plt.subplot(131)
    plt.plot(m.t.value, x, 'ro', label='x')
    plt.title('State Soln')
    plt.xlabel('time')
    plt.subplot(132)
    plt.plot(m.t.value, u, 'ro', label='u')
    plt.title('Control Soln')
    plt.xlabel('time')
    plt.subplot(133)
    plt.plot(m.t.value, F, 'ro', label='Cost Integrand')
    plt.title('Anti-derivative of \n Cost Integrand')
    plt.xlabel('time')
    return plt