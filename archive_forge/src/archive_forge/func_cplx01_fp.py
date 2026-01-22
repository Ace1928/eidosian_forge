from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def cplx01_fp(z, n, a):
    return n * z ** (n - 1)