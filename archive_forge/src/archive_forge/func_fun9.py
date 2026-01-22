from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def fun9(x):
    return np.exp(x) - 2 - 0.01 / x ** 2 + 2e-06 / x ** 3