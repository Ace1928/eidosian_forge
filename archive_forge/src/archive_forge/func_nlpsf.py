import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def nlpsf(free_params, data=data):
    with np.errstate(invalid='ignore', divide='ignore'):
        return dist._penalized_nlpsf(free_params, data)