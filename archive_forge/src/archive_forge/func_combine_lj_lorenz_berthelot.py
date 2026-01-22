import numpy as np
import ase.units as units
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_masses
from ase.geometry import find_mic
def combine_lj_lorenz_berthelot(sigma, epsilon):
    """Combine LJ parameters according to the Lorenz-Berthelot rule"""
    sigma_c = np.zeros((len(sigma), len(sigma)))
    epsilon_c = np.zeros_like(sigma_c)
    for ii in range(len(sigma)):
        sigma_c[:, ii] = (sigma[ii] + sigma) / 2
        epsilon_c[:, ii] = (epsilon[ii] * epsilon) ** 0.5
    return (sigma_c, epsilon_c)