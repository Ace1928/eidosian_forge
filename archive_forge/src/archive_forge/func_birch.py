import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def birch(V, E0, B0, BP, V0):
    """
    From Intermetallic compounds: Principles and Practice, Vol. I: Principles
    Chapter 9 pages 195-210 by M. Mehl. B. Klein, D. Papaconstantopoulos
    paper downloaded from Web

    case where n=0
    """
    E = E0 + 9 / 8 * B0 * V0 * ((V0 / V) ** (2 / 3) - 1) ** 2 + 9 / 16 * B0 * V0 * (BP - 4) * ((V0 / V) ** (2 / 3) - 1) ** 3
    return E