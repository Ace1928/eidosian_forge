from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def cplx01_fpp(z, n, a):
    return n * (n - 1) * z ** (n - 2)