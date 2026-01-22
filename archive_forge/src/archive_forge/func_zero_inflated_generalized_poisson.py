import os
import numpy as np
def zero_inflated_generalized_poisson():
    obj = Namespace()
    obj.nobs = 20190
    obj.params = [3.57337, -17.95797, -0.2138, 0.03847, -0.05348, 1.15666, 1.36468]
    obj.llf = -43630.6
    obj.bse = [1.66109, 7.62052, 0.02066, 0.00339, 0.00289, 0.0168, 0.01606]
    obj.aic = 87275
    return obj