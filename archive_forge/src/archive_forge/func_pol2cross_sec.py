import numpy as np
import ase.units as un
def pol2cross_sec(p, omg):
    """
    Convert the polarizability in au to cross section in nm**2

    Input parameters:
    -----------------
    p (np array): polarizability from mbpt_lcao calc
    omg (np.array): frequency range in eV

    Output parameters:
    ------------------
    sigma (np array): cross section in nm**2
    """
    c = 1 / un.alpha
    omg /= un.Ha
    sigma = 4 * np.pi * omg * p / c
    return sigma * (0.1 * un.Bohr) ** 2