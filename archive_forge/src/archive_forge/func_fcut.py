import numpy as np
from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list
def fcut(r, r0, r1):
    """
    Piecewise quintic C^{2,1} regular polynomial for use as a smooth cutoff.
    Ported from JuLIP.jl, https://github.com/JuliaMolSim/JuLIP.jl
    
    Parameters
    ----------
    r0 - inner cutoff radius
    r1 - outder cutoff radius
    """
    s = 1.0 - (r - r0) / (r1 - r0)
    return (s >= 1.0) + ((0.0 < s) & (s < 1.0)) * (6.0 * s ** 5 - 15.0 * s ** 4 + 10.0 * s ** 3)