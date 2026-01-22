import numpy as np
from scipy import optimize
from statsmodels.base.model import Model
def error2(params, x, y):
    return (y - func(params, x)) ** 2