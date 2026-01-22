import numpy as np
from scipy import optimize
from statsmodels.base.model import Model
jacobian of prediction function using complex step derivative

        This assumes that the predict function does not use complex variable
        but is designed to do so.

        