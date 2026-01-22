import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def p3(V, c0, c1, c2, c3):
    """polynomial fit"""
    E = c0 + c1 * V + c2 * V ** 2 + c3 * V ** 3
    return E