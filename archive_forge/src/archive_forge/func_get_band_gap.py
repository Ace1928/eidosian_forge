import functools
import warnings
import numpy as np
from ase.utils import IOContext
def get_band_gap(calc, direct=False, spin=None, output='-'):
    warnings.warn('Please use ase.dft.bandgap.bandgap() instead!')
    gap, (s1, k1, n1), (s2, k2, n2) = bandgap(calc, direct, spin, output)
    ns = calc.get_number_of_spins()
    if ns == 2 and spin is None:
        return (gap, (s1, k1), (s2, k2))
    return (gap, k1, k2)