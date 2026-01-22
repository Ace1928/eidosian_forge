import numpy as np
from statsmodels.genmod.families import Poisson
from .gee_gaussian_simulation_check import GEE_simulator
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable,Independence
def gendat_exchangeable():
    exs = Exchangeable_simulator()
    exs.params = np.r_[2.0, 0.2, 0.2, -0.1, -0.2]
    exs.ngroups = 200
    exs.dparams = [0.3]
    exs.simulate()
    return (exs, Exchangeable())