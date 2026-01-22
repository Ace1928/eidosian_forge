import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize, Bounds
from scipy.special import gammaln
from scipy._lib._util import check_random_state
from scipy.optimize._constraints import new_bounds_to_old
def update_best(self, e, x, context):
    self.ebest = e
    self.xbest = np.copy(x)
    if self.callback is not None:
        val = self.callback(x, e, context)
        if val is not None:
            if val:
                return 'Callback function requested to stop early by returning True'