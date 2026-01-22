import numpy as np
import warnings
from scipy.optimize import minimize
from ase.parallel import world
from ase.io.jsonio import write_json
from ase.optimize.optimize import Optimizer
from ase.optimize.gpmin.gp import GaussianProcess
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ConstantPrior
def relax_model(self, r0):
    result = minimize(self.acquisition, r0, method='L-BFGS-B', jac=True)
    if result.success:
        return result.x
    else:
        self.dump()
        raise RuntimeError('The minimization of the acquisition function has not converged')