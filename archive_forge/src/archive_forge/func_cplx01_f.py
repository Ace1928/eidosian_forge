from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def cplx01_f(z, n, a):
    """z**n-a:  Use to find the nth root of a"""
    return z ** n - a